"""
OhioT1DM dataset loader.

The OhioT1DM dataset contains CGM data from 12 patients with Type 1 diabetes.
Data is stored in XML format with glucose readings at 5-minute intervals.
"""

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


class OhioT1DMLoader:
    """
    Loader for OhioT1DM dataset.

    Dataset structure:
    - 12 patients (IDs: 559, 563, 570, 575, 588, 591, 540, 544, 552, 567, 584, 596)
    - ~8 weeks of data per patient
    - 5-minute CGM sampling interval
    - Includes: glucose, meals, insulin, exercise, sleep
    """

    PATIENT_IDS = [559, 563, 570, 575, 588, 591, 540, 544, 552, 567, 584, 596]
    DATE_FORMAT = "%d-%m-%Y %H:%M:%S"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_patient(self, patient_id: int, file_type: str = "training") -> Dict[str, pd.DataFrame]:
        """
        Load all data for a single patient.

        Args:
            patient_id: Patient ID (e.g., 559, 563, etc.)
            file_type: "training" or "testing"

        Returns:
            Dictionary with 'glucose', 'meals', 'insulin', 'basal', 'exercise' DataFrames
        """
        xml_file = self.data_dir / f"{patient_id}-ws-{file_type}.xml"

        if not xml_file.exists():
            raise FileNotFoundError(f"Patient file not found: {xml_file}")

        tree = ET.parse(xml_file)
        root = tree.getroot()

        result = {
            'glucose': self._parse_glucose(root),
            'meals': self._parse_meals(root),
            'bolus': self._parse_bolus(root),
            'basal': self._parse_basal(root),
            'exercise': self._parse_exercise(root),
            'sleep': self._parse_sleep(root),
            'finger_stick': self._parse_finger_stick(root),
        }

        return result

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp from OhioT1DM format."""
        return datetime.strptime(ts_str, self.DATE_FORMAT)

    def _parse_glucose(self, root: ET.Element) -> pd.DataFrame:
        """Parse glucose CGM readings from all glucose_level segments."""
        records = []

        for glucose_level in root.findall('glucose_level'):
            for event in glucose_level.findall('event'):
                ts = event.get('ts')
                value = event.get('value')
                if ts and value:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'glucose': float(value)
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        return df

    def _parse_meals(self, root: ET.Element) -> pd.DataFrame:
        """Parse meal/carbohydrate entries."""
        records = []

        meal_elem = root.find('meal')
        if meal_elem is not None:
            for event in meal_elem.findall('event'):
                ts = event.get('ts')
                carbs = event.get('carbs')
                meal_type = event.get('type', 'unknown')
                if ts:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'carbs': float(carbs) if carbs else 0.0,
                        'type': meal_type
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_bolus(self, root: ET.Element) -> pd.DataFrame:
        """Parse bolus insulin entries."""
        records = []

        bolus_elem = root.find('bolus')
        if bolus_elem is not None:
            for event in bolus_elem.findall('event'):
                ts_begin = event.get('ts_begin')
                ts_end = event.get('ts_end')
                dose = event.get('dose')
                bolus_type = event.get('type', 'normal')
                bwz_carb = event.get('bwz_carb_input')

                if ts_begin:
                    records.append({
                        'timestamp': self._parse_timestamp(ts_begin),
                        'ts_end': self._parse_timestamp(ts_end) if ts_end else None,
                        'dose': float(dose) if dose else 0.0,
                        'type': bolus_type,
                        'carb_input': float(bwz_carb) if bwz_carb else 0.0
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_basal(self, root: ET.Element) -> pd.DataFrame:
        """Parse basal insulin rate entries."""
        records = []

        basal_elem = root.find('basal')
        if basal_elem is not None:
            for event in basal_elem.findall('event'):
                ts = event.get('ts')
                value = event.get('value')
                if ts:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'rate': float(value) if value else 0.0
                    })

        # Also parse temp_basal if exists
        temp_basal_elem = root.find('temp_basal')
        if temp_basal_elem is not None:
            for event in temp_basal_elem.findall('event'):
                ts_begin = event.get('ts_begin')
                ts_end = event.get('ts_end')
                value = event.get('value')
                if ts_begin:
                    records.append({
                        'timestamp': self._parse_timestamp(ts_begin),
                        'rate': float(value) if value else 0.0,
                        'is_temp': True
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_exercise(self, root: ET.Element) -> pd.DataFrame:
        """Parse exercise entries."""
        records = []

        exercise_elem = root.find('exercise')
        if exercise_elem is not None:
            for event in exercise_elem.findall('event'):
                ts = event.get('ts')
                intensity = event.get('intensity')
                duration = event.get('duration')
                if ts:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'intensity': int(intensity) if intensity else 0,
                        'duration': int(duration) if duration else 0
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_sleep(self, root: ET.Element) -> pd.DataFrame:
        """Parse sleep entries."""
        records = []

        sleep_elem = root.find('sleep')
        if sleep_elem is not None:
            for event in sleep_elem.findall('event'):
                ts = event.get('ts')
                quality = event.get('quality')
                if ts:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'quality': int(quality) if quality else 0
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def _parse_finger_stick(self, root: ET.Element) -> pd.DataFrame:
        """Parse finger stick blood glucose readings."""
        records = []

        fs_elem = root.find('finger_stick')
        if fs_elem is not None:
            for event in fs_elem.findall('event'):
                ts = event.get('ts')
                value = event.get('value')
                if ts and value:
                    records.append({
                        'timestamp': self._parse_timestamp(ts),
                        'glucose': float(value)
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def load_all_patients(self, file_type: str = "training") -> Dict[int, Dict[str, pd.DataFrame]]:
        """Load data for all available patients."""
        all_data = {}

        for pid in self.PATIENT_IDS:
            try:
                all_data[pid] = self.load_patient(pid, file_type)
                n_glucose = len(all_data[pid]['glucose'])
                n_meals = len(all_data[pid]['meals'])
                n_bolus = len(all_data[pid]['bolus'])
                print(f"Loaded patient {pid}: {n_glucose} glucose, {n_meals} meals, {n_bolus} bolus")
            except FileNotFoundError as e:
                print(f"Warning: Could not load patient {pid}: {e}")
            except Exception as e:
                print(f"Error loading patient {pid}: {e}")

        return all_data

    def get_available_patients(self) -> List[int]:
        """Get list of patient IDs with available data files."""
        available = []
        for pid in self.PATIENT_IDS:
            training_file = self.data_dir / f"{pid}-ws-training.xml"
            if training_file.exists():
                available.append(pid)
        return available


def prepare_patient_data(
    patient_data: Dict[str, pd.DataFrame],
    resample_interval: str = '5min'
) -> pd.DataFrame:
    """
    Prepare patient data for modeling.

    - Resamples to regular 5-minute intervals
    - Handles missing values
    - Merges auxiliary signals (meals, insulin)

    Args:
        patient_data: Dictionary from OhioT1DMLoader.load_patient()
        resample_interval: Resampling interval (default: '5min')

    Returns:
        DataFrame with timestamp index and glucose + auxiliary features
    """
    glucose_df = patient_data['glucose'].copy()

    if glucose_df.empty:
        return glucose_df

    # Set timestamp as index
    glucose_df = glucose_df.set_index('timestamp')

    # Create full time range at 5-minute intervals
    start_time = glucose_df.index.min()
    end_time = glucose_df.index.max()
    full_range = pd.date_range(start=start_time, end=end_time, freq=resample_interval)

    # Reindex to full range
    glucose_df = glucose_df.reindex(full_range)
    glucose_df.index.name = 'timestamp'

    # Interpolate small gaps (up to 30 minutes = 6 samples)
    glucose_df['glucose'] = glucose_df['glucose'].interpolate(
        method='linear', limit=6
    )

    # Add meal carbs as a feature (0 when no meal)
    meals_df = patient_data['meals'].copy()
    if not meals_df.empty:
        meals_df = meals_df.set_index('timestamp')
        meals_df = meals_df.reindex(full_range, fill_value=0)
        glucose_df['carbs'] = meals_df['carbs'].fillna(0)
    else:
        glucose_df['carbs'] = 0.0

    # Add bolus insulin as a feature
    bolus_df = patient_data['bolus'].copy()
    if not bolus_df.empty:
        bolus_df = bolus_df.set_index('timestamp')
        bolus_df = bolus_df.reindex(full_range, fill_value=0)
        glucose_df['bolus'] = bolus_df['dose'].fillna(0)
    else:
        glucose_df['bolus'] = 0.0

    return glucose_df.reset_index()


def get_patient_stats(patient_data: Dict[str, pd.DataFrame]) -> Dict:
    """Get summary statistics for a patient's data."""
    glucose_df = patient_data['glucose']

    if glucose_df.empty:
        return {}

    glucose_values = glucose_df['glucose'].values

    stats = {
        'n_readings': len(glucose_df),
        'duration_days': (glucose_df['timestamp'].max() - glucose_df['timestamp'].min()).days,
        'mean_glucose': np.mean(glucose_values),
        'std_glucose': np.std(glucose_values),
        'min_glucose': np.min(glucose_values),
        'max_glucose': np.max(glucose_values),
        'time_in_range': np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100,
        'time_below_70': np.mean(glucose_values < 70) * 100,
        'time_above_180': np.mean(glucose_values > 180) * 100,
        'n_meals': len(patient_data['meals']),
        'n_bolus': len(patient_data['bolus']),
    }

    return stats
