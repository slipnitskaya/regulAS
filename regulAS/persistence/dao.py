from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.orm.relationships import RelationshipProperty

from sqlalchemy import Column, Float, String, Integer, DateTime, ForeignKey


class BaseTable(object):

    def __repr__(self):
        fields = list()
        for name, value in vars(self.__class__).items():
            if not name.startswith('_'):
                if isinstance(value.property, RelationshipProperty):
                    field_value = 'List<{}>' if value.property.uselist else '{}'
                    field_value = field_value.format(value.property.argument)
                else:
                    field_value = getattr(self, name)

                fields.append(f'{name}={field_value}')

        return '<{}({})>'.format(self.__class__.__name__, ', '.join(fields))


RegulASTable = declarative_base(cls=BaseTable)


class Data(RegulASTable):
    __tablename__ = 'Data'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    name = Column('name', String(256))
    meta = Column('meta', String(4096))
    num_samples = Column('num_samples', Integer)
    num_features = Column('num_features', Integer)
    md5 = Column('md5', String(32), unique=True)

    experiments = relationship('Experiment', back_populates='data')


class Pipeline(RegulASTable):
    __tablename__ = 'Pipeline'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    transformation_idx = Column('transformation_id', ForeignKey('Transformation.ID'))
    model_idx = Column('model_id', ForeignKey('Model.ID'))

    model = relationship('Model', back_populates='pipelines')
    transformation = relationship('Transformation', back_populates='pipelines')
    experiment = relationship('Experiment', back_populates='pipeline')


class Transformation(RegulASTable):
    __tablename__ = 'Transformation'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    prev_idx = Column('prev_id', ForeignKey('Transformation.ID'))
    source_idx = Column('source_id', ForeignKey('TransformationSource.ID'))
    parameters = Column('parameters', String(16384))

    pipelines = relationship('Pipeline', back_populates='transformation')


class TransformationSource(RegulASTable):
    __tablename__ = 'TransformationSource'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    fqn = Column('fqn', String(512))
    version = Column('version', String(128))
    source = Column('source', String(32768))


class Model(RegulASTable):
    __tablename__ = 'Model'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    fqn = Column('fqn', String(512))
    version = Column('version', String(128))
    source = Column('source', String(32768))

    pipelines = relationship('Pipeline', back_populates='model')
    hyper_parameters = relationship('HyperParameter', back_populates='model')


class HyperParameter(RegulASTable):
    __tablename__ = 'HyperParameter'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    model_idx = Column('model_id', ForeignKey('Model.ID'))
    name = Column('name', String(512))

    model = relationship('Model', back_populates='hyper_parameters')


class HyperParameterValue(RegulASTable):
    __tablename__ = 'HyperParameterValue'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    hyper_parameter_idx = Column('hyper_parameter_id', ForeignKey('HyperParameter.ID'))
    pipeline_idx = Column('pipeline_id', ForeignKey('Pipeline.ID'))
    value = Column('value', String(128))


class FeatureRanking(RegulASTable):
    __tablename__ = 'FeatureRanking'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    experiment_idx = Column('experiment_id', ForeignKey('Experiment.ID'))
    feature = Column('feature', String(128))
    score = Column('score', Float)

    experiment = relationship('Experiment', back_populates='feature_scores')


class Prediction(RegulASTable):
    __tablename__ = 'Prediction'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    experiment_idx = Column('experiment_id', ForeignKey('Experiment.ID'))
    sample_name = Column('sample_name', String(128))
    true_value = Column('true_value', Float)
    predicted_value = Column('predicted_value', Float)

    experiment = relationship('Experiment', back_populates='predictions')


class Experiment(RegulASTable):
    __tablename__ = 'Experiment'

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    timestamp = Column('timestamp', DateTime)
    name = Column('name', String(1024))
    data_idx = Column('data_id', ForeignKey('Data.ID'))
    pipeline_idx = Column('pipeline_id', ForeignKey('Pipeline.ID'))
    config = Column('config', String(65536))

    data = relationship('Data', back_populates='experiments')
    pipeline = relationship('Pipeline', back_populates='experiment')
    feature_scores = relationship('FeatureRanking', back_populates='experiment')
    predictions = relationship('Prediction', back_populates='experiment')
