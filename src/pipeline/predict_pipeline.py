import sys
from pathlib import Path
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, input_df: pd.DataFrame):
        """Run preprocessing and model prediction on input dataframe.

        input_df should contain the same feature columns used during training.
        """
        try:
            project_root = Path(__file__).resolve().parents[2]
            model_path = str(project_root / 'artifacts' / 'model.pkl')
            preprocessor_path = str(project_root / 'artifacts' / 'preprocessor.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(input_df)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self) -> pd.DataFrame:
        try:
            custom_data_input = {
                'gender': [self.gender],
                'race/ethnicity': [self.race_ethnicity],
                'parental level of education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test preparation course': [self.test_preparation_course],
                'reading score': [self.reading_score],
                'writing score': [self.writing_score],
            }

            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)