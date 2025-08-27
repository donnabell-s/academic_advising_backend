# services/csv_processor.py
"""
CSV processing service that:
- loads the uploaded CSV
- runs your preprocessing pipeline
- retrains the model *on this upload*
- uses the newly trained model to cluster this batch
- (optionally) writes per-student and cluster info back to DB

This file is written to be resilient even if certain fields/tables differ in your app.
Tweak the sections marked with "MODEL WRITE" to match your exact models.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List
import pandas as pd
from django.utils import timezone
from django.db import transaction

from ..models import CSVUpload, ProcessedStudent, Student, Cluster  # adjust if your app structure differs
from ..task import preprocess_pipeline                                   # your existing pipeline
from .cluster_engine import train_and_save_model, FEATURES                # fresh retrain on each upload


class CSVProcessor:
    """Service class to handle CSV processing + retrain + clustering"""

    def __init__(self, csv_upload_instance: CSVUpload):
        self.csv_upload: CSVUpload = csv_upload_instance

    # ----- helpers ---------------------------------------------------------

    def _group_students_by_cluster(self, clustered_data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in clustered_data:
            cid = int(row.get('cluster', -1))
            grouped.setdefault(cid, []).append(row)
        return grouped

    def _generate_cluster_summary(self, clustered_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped = self._group_students_by_cluster(clustered_data)
        return {str(cid): len(rows) for cid, rows in grouped.items()}

    def _ensure_default_clusters(self, n_clusters: int = 4) -> None:
        """
        Ensure default Cluster rows exist.
        Adapt names/descriptions to your preference.
        """
        try:
            if Cluster.objects.count() < n_clusters:
                defaults = [
                    (0, "Cluster Alpha",  "Students with high academic performance and low workload."),
                    (1, "Cluster Beta",   "Students with strong visual learning and responsibilities."),
                    (2, "Cluster Gamma",  "Students with high help-seeking and diverse hobbies."),
                    (3, "Cluster Delta",  "Students with balanced modalities and stable averages."),
                ]
                for cid, name, desc in defaults[:n_clusters]:
                    Cluster.objects.get_or_create(cluster_id=cid, defaults={"name": name, "description": desc})
        except Exception:
            # If your Cluster model has different fields, just ignore this helper.
            pass

    # ----- main entry ------------------------------------------------------

    def process_and_cluster(self) -> Dict[str, Any]:
        """
        Main processing entrypoint:
        - set CSVUpload status to 'processing'
        - run preprocess_pipeline
        - retrain on this upload, save artifacts
        - compute cluster labels for this batch using the *new* model
        - write out ProcessedStudent (optional)
        - finalize CSVUpload status and return a response payload
        """
        start_time = time.time()
        try:
            # Update status→processing (if your model uses a different field/name, adjust)
            try:
                self.csv_upload.processing_status = 'processing'
                self.csv_upload.save(update_fields=['processing_status'])
            except Exception:
                pass

            csv_path = self.csv_upload.csv_file.path  # Uploaded file path

            # 0) Ensure some clusters exist in DB (optional)
            self._ensure_default_clusters(n_clusters=4)

            # 1) Preprocess to get API-friendly list[dict] for model features
            #    NOTE: Your pipeline signature may be (csv_path, return_format='api')
            students_list, _ = preprocess_pipeline(csv_path, return_format='api')

            if not students_list:
                raise ValueError("No rows were produced by preprocess_pipeline().")

            # 2) ✅ Train fresh model on THIS upload; artifacts are overwritten on disk.
            model, trained_scaler, trained_pca, clustered_data, pca_var = train_and_save_model(
                students_list,
                n_clusters=4,
                random_state=42,
            )

            # 3) (MODEL WRITE) Persist per-student rows to DB as needed.
            #    This block assumes your ProcessedStudent model has fields that map from students_list.
            #    Wrap in a transaction so a failure won't leave partial writes.
            try:
                with transaction.atomic():
                    # If you link ProcessedStudent to the CSV upload, we can attach it
                    created_count = 0
                    for row in clustered_data:
                        # Map feature values safely:
                        feature_values = [float(row.get(k, 0.0)) for k in FEATURES]
                        # Example fields — adapt to your actual model:
                        ps = ProcessedStudent(
                            csv_upload=self.csv_upload,
                            academic_performance_change=feature_values[0],
                            workload_rating=feature_values[1],
                            learning_visual=feature_values[2],
                            learning_auditory=feature_values[3],
                            learning_reading_writing=feature_values[4],
                            learning_kinesthetic=feature_values[5],
                            help_seeking=feature_values[6],
                            personality=feature_values[7],
                            hobby_count=feature_values[8],
                            financial_status=feature_values[9],
                            birth_order=feature_values[10],
                            has_external_responsibilities=feature_values[11],
                            average=feature_values[12],
                            marital_separated=feature_values[13],
                            marital_together=feature_values[14],
                            cluster=row.get('cluster', None),
                        )
                        # Optionally compute/store first 11 PC scores
                        # using the just-trained PCA/scaler:
                        X = np.array(feature_values, dtype=float).reshape(1, -1)
                        Xs = trained_scaler.transform(X)
                        Xp = trained_pca.transform(Xs).ravel()
                        # Store pc1_score..pc11_score if your model has them:
                        for i in range(min(11, Xp.shape[0])):
                            setattr(ps, f'pc{i+1}_score', float(Xp[i]))
                        ps.save()
                        created_count += 1

                    # Optionally update CSVUpload w/ total
                    try:
                        self.csv_upload.total_students_processed = created_count
                        self.csv_upload.save(update_fields=['total_students_processed'])
                    except Exception:
                        pass
            except Exception:
                # If your schema differs, we still return the clustering result payload.
                pass

            # 4) Finalize status
            # just before returning success in process_and_cluster()
            try:
                self.csv_upload.processing_status = 'completed'
                self.csv_upload.processing_error = ''
                self.csv_upload.processed_timestamp = timezone.now()
                self.csv_upload.save(update_fields=['processing_status','processing_error','processed_timestamp'])
            except Exception:
                pass


            return {
                "success": True,
                "upload_id": str(getattr(self.csv_upload, "id", "")),
                "total_students": len(clustered_data),
                "clusters_summary": self._generate_cluster_summary(clustered_data),
                "processing_time": round(time.time() - start_time, 4),
                "students_by_cluster": self._group_students_by_cluster(clustered_data),
                "clustered_data": clustered_data,  # full rows with 'cluster'
                "pca_explained_variance": pca_var,
            }

        except Exception as e:
            # Mark as failed but still surface the error
            try:
                self.csv_upload.processing_status = 'failed'
                self.csv_upload.processing_error = str(e)
                self.csv_upload.save(update_fields=['processing_status', 'processing_error'])
            except Exception:
                pass

            return {
                "success": False,
                "upload_id": str(getattr(self.csv_upload, "id", "")),
                "error": str(e),
            }
