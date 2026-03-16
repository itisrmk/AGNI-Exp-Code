# OhioT1DM Dataset - Complete Structure Analysis

## Dataset Overview

**Source**: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html (requires Data Use Agreement)
**Alternative Access**: https://www.kaggle.com/datasets/ryanmouton/ohiot1dm (public, MIT License)

**Dataset Size**: 2.3 MB (compressed ZIP)
**Format**: XML files
**Number of Subjects**: 12 patients with Type 1 Diabetes
**Duration**: 8 weeks per patient
**Sampling Frequency**: 5 minutes for CGM readings

## File Structure

### Files Per Patient
Each patient has 2 XML files:
- **Training/Development file**: First portion of data for model training
- **Testing file**: Held-out portion for evaluation

**Total files**: 24 XML files (12 patients × 2 files each)

### Patient Cohorts
- **2018 Cohort**: 6 patients (original release)
- **2020 Cohort**: 6 additional patients (extended release)

## XML Schema Structure

Based on code analysis from https://github.com/yixiangD/AccurateBG/blob/master/accurate_bg/data_reader.py

### Root Structure
```xml
<root>
    <glucose_level>
        <!-- Multiple glucose entries -->
    </glucose_level>
    <glucose_level>
        <!-- Another continuous segment -->
    </glucose_level>
    ...
    <basal>
        <!-- Basal insulin entries -->
    </basal>
    <bolus>
        <!-- Bolus insulin entries -->
    </bolus>
    <meal>
        <!-- Meal/carbohydrate entries -->
    </meal>
    <sleep>
        <!-- Sleep quality entries -->
    </sleep>
    <work>
        <!-- Work/activity entries -->
    </work>
    <exercise>
        <!-- Exercise entries -->
    </exercise>
    <stress>
        <!-- Stress events -->
    </stress>
    <illness>
        <!-- Illness events -->
    </illness>
</root>
```

## Detailed Element Specifications

### 1. `<glucose_level>` - CGM Readings

Each `<glucose_level>` element contains a continuous segment of CGM readings.

**Child elements**: Multiple entries with attributes:
```xml
<glucose_level>
    <event ts="01-01-2024 00:00:00" value="120.5" />
    <event ts="01-01-2024 00:05:00" value="118.2" />
    <event ts="01-01-2024 00:10:00" value="115.8" />
    ...
</glucose_level>
```

**Attributes**:
- `ts` (timestamp): Format "%d-%m-%Y %H:%M:%S" (e.g., "01-01-2024 14:30:00")
- `value` (float): Blood glucose level in mg/dL

**Characteristics**:
- 5-minute resolution (12 readings per hour, 288 per day)
- Multiple `<glucose_level>` segments per file (discontinuous recordings)
- 8 weeks × 7 days × 288 readings = ~16,128 expected readings per patient

### 2. `<basal>` - Basal Insulin

Temporary basal insulin rate changes that supersede the patient's normal basal rate.

**Attributes**:
- `ts` (timestamp): When the basal rate change occurred
- `value` (float): Insulin rate in Units/hour
  - `value="0"`: Basal insulin flow suspended
  - At end of temp basal: Rate returns to normal

**Example**:
```xml
<basal>
    <event ts="01-01-2024 08:00:00" value="1.2" />
    <event ts="01-01-2024 12:00:00" value="0.0" />  <!-- Suspended -->
</basal>
```

### 3. `<bolus>` - Bolus Insulin

Insulin delivered to the patient, typically before meals or when hyperglycemic.

**Attributes**:
- `ts` (timestamp): When bolus was administered
- `dose` (float): Amount of insulin in Units
- `type` (string): Delivery type
  - "normal": All insulin delivered at once (most common)
  - Other types: Extended delivery patterns

**Example**:
```xml
<bolus>
    <event ts="01-01-2024 12:00:00" dose="8.5" type="normal" />
    <event ts="01-01-2024 18:30:00" dose="6.2" type="normal" />
</bolus>
```

### 4. `<meal>` - Carbohydrate Intake

Self-reported meal times and carbohydrate estimates.

**Attributes**:
- `ts` (timestamp): When meal was consumed
- `carbs` (float): Carbohydrate estimate in grams
- `type` (string): Meal type (e.g., "breakfast", "lunch", "dinner", "snack")

**Example**:
```xml
<meal>
    <event ts="01-01-2024 08:00:00" carbs="45.0" type="breakfast" />
    <event ts="01-01-2024 12:30:00" carbs="60.0" type="lunch" />
</meal>
```

### 5. `<sleep>` - Sleep Events

Self-reported sleep times and quality assessment.

**Attributes**:
- `ts` (timestamp): Sleep event time (start or end)
- `quality` (int): Subjective sleep quality
  - 1 = Poor
  - 2 = Fair
  - 3 = Good

**Example**:
```xml
<sleep>
    <event ts="01-01-2024 23:00:00" quality="3" />
    <event ts="02-01-2024 07:00:00" quality="3" />
</sleep>
```

### 6. `<work>` - Work Events

Self-reported work times with physical exertion intensity.

**Attributes**:
- `ts` (timestamp): Work event time (arrival/departure)
- `intensity` (int): Physical exertion level (1-10 scale)

**Example**:
```xml
<work>
    <event ts="01-01-2024 09:00:00" intensity="5" />
    <event ts="01-01-2024 17:00:00" intensity="5" />
</work>
```

### 7. `<exercise>` - Exercise Events

Self-reported exercise times and intensity.

**Attributes**:
- `ts` (timestamp): When exercise occurred
- `intensity` (int): Intensity level
- `duration` (int): Duration in minutes (likely)

### 8. `<stress>` - Stress Events

Self-reported stress events.

**Attributes**:
- `ts` (timestamp): When stress was experienced
- `level` (int): Stress intensity level

### 9. `<illness>` - Illness Events

Self-reported illness occurrences.

**Attributes**:
- `ts` (timestamp): When illness was reported
- `type` (string): Type/description of illness

## Data Characteristics

### Temporal Properties
- **Resolution**: 5 minutes for CGM
- **Duration**: 8 weeks continuous monitoring
- **Date Shifting**: All dates shifted by same random amount (removes seasonality)

### Missing Data
- **Patient Weights**: Unavailable (placeholder value 99 used)
- **Gaps**: CGM data may have gaps (captured in separate `<glucose_level>` segments)

### Privacy
- **De-identification**: HIPAA Safe Harbor compliant
- **Access**: Requires Data Use Agreement (DUA) from official source

### Physiological Sensors (Additional Data)
Beyond the core XML, the dataset includes:
- **Basis Peak sensor**: Heart rate, galvanic skin response, skin temperature, air temperature, step count, acceleration (5-min aggregations)
- **Empatica Embrace band**: Similar physiological metrics (1-min aggregations)

## Code Example: Reading OhioT1DM Data

Based on AccurateBG repository:

```python
import xml.etree.ElementTree as ET
import datetime

def read_ohio(filepath):
    """Read OhioT1DM XML file and extract glucose readings"""
    tree = ET.parse(filepath)
    root = tree.getroot()

    glucose_segments = []
    for item in root.findall("glucose_level"):
        segment = []
        for entry in item:
            ts = datetime.datetime.strptime(entry.attrib["ts"], "%d-%m-%Y %H:%M:%S")
            value = float(entry.attrib["value"])
            segment.append((ts, value))
        glucose_segments.append(segment)

    return glucose_segments
```

## Expected Data Volume

### Per Patient (8 weeks)
- **CGM Readings**: ~16,128 records (56 days × 288 readings/day)
- **Meals**: ~168-224 records (3-4 meals/day × 56 days)
- **Insulin Boluses**: ~168-224 records (aligned with meals)
- **Basal Changes**: Variable (patient-specific)
- **Sleep Events**: ~112 records (2 events/day × 56 days)
- **Other Events**: Variable based on patient lifestyle

### Total Dataset (12 Patients)
- **CGM Readings**: ~193,536 records
- **Total Events**: ~250,000+ including all event types

## Data Quality Considerations

### Advantages
✓ Real-world clinical data
✓ Authentic physiological variability
✓ Multiple input modalities (glucose, insulin, meals, lifestyle)
✓ Standardized XML format
✓ Well-documented and widely used in research

### Limitations
⚠️ Small sample size (12 patients)
⚠️ Limited to Type 1 Diabetes patients on insulin pumps
⚠️ Self-reported data may have accuracy issues
⚠️ Missing patient demographics (weights removed)
⚠️ Date shifting removes temporal patterns (seasonality, holidays)

## Use for Continual Learning Research

### Suitability for CONGA Project
✓ **5-minute resolution**: Matches proposal requirements
✓ **8-week duration**: Sufficient for train/test temporal splits
✓ **Auxiliary inputs**: Carbs and insulin available as specified
✓ **Temporal drift**: 8 weeks captures physiological changes
✓ **Established benchmark**: Enables comparison with prior work

### Recommended Usage
1. **Per-patient models**: 12 independent experiments (within-subject)
2. **Temporal split**: First 4 weeks training, last 4 weeks testing
3. **Features**: CGM history, carbs, insulin (all in XML)
4. **Prediction horizons**: 15, 30, 60 minutes (3, 6, 12 steps)

## References

- **Original Paper**: Marling & Bunescu (2020). "The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020"
- **Official Site**: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
- **Kaggle Mirror**: https://www.kaggle.com/datasets/ryanmouton/ohiot1dm
- **Code Examples**: https://github.com/topics/ohiot1dm

---

**Last Updated**: October 27, 2025
**Analysis Source**: Code inspection of AccurateBG repository + Web research
