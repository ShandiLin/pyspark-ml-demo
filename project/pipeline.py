from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark import keyword_only
from pyspark.ml.util import MLReadable, MLWritable, Identifiable


from pyspark_pipeline_wrapper import PysparkReaderWriter


class DropCols(Transformer, Identifiable, PysparkReaderWriter, MLReadable, MLWritable):

    def __init__(self, cols=None):
        super(DropCols, self).__init__()
        self.cols = cols

    def _transform(self, df):
        return df.drop(*self.cols)


class Str2Map(Transformer, HasInputCol, HasOutputCol, Identifiable, PysparkReaderWriter, MLReadable, MLWritable):

    @keyword_only
    def __init__(self, maps=None, inputCol=None, outputCol=None):
        super(Str2Map, self).__init__()
        self.maps = Param(self, "maps", "")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, maps, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setMaps(self, value):
        self._paramMap[self.maps] = value
        return self

    def getMaps(self):
        return self.getOrDefault(self.maps)

    def _transform(self, df):
        import pyspark.sql.functions as F

        maps = self.getMaps()
        tt = F.udf(lambda v: maps.get(v))(df[self.getInputCol()])
        return df.withColumn(self.getOutputCol(), tt.cast('int'))


def get_stages(df):
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    # for serving code to call above class
    from pipeline import DropCols, Str2Map

    dropOver18 = DropCols(['over18'])
    overtime = Str2Map(maps={'No': 0, 'Yes': 1},
                       inputCol='OverTime', outputCol='OverTime')
    gender = Str2Map(maps={'Male': 0, 'Female': 1},
                     inputCol='Gender', outputCol='Gender')
    marital = Str2Map(maps={'Married': 2, 'Divorced': 1, 'Single': 0},
                      inputCol='MaritalStatus', outputCol='MaritalStatus')
    bsTravel = Str2Map(maps={'Travel_Frequently': 2, 'Non-Travel': 0,
                             'Travel_Rarely': 1}, inputCol='BusinessTravel', outputCol='BusinessTravel')
    dropStrIdxer = DropCols(["Department", "EducationField", "JobRole"])
    dept = StringIndexer(inputCol="Department", outputCol="Department_Idx")
    edu_field = StringIndexer(inputCol="EducationField",
                              outputCol="EducationField_Idx")
    job_role = StringIndexer(inputCol="JobRole", outputCol="JobRole_Idx")

    features = ['Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
                'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'Department_Idx', 'EducationField_Idx', 'JobRole_Idx']
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    dropOther = DropCols(features)

    stages = [dropOver18, overtime, gender, marital,
              bsTravel, dept, edu_field, job_role,
              dropStrIdxer, assembler, dropOther]
    return stages


def create_pipeline(stages):
    from pyspark.ml import Pipeline
    return Pipeline(stages=stages)
