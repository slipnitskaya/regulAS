import os
import enum

from sqlalchemy.engine.url import URL
from sqlalchemy.sql.functions import now
from sqlalchemy.orm.relationships import RelationshipProperty
from sqlalchemy.orm import Session, relationship, declarative_base

from sqlalchemy import Enum, Float, Column, String, Boolean, Integer, DateTime, ForeignKey


class BaseTable(object):

    idx = Column('ID', Integer, primary_key=True, autoincrement=True)
    time_created = Column('time_created', DateTime, nullable=False, default=now())

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


def create_schema(url: URL, engine: Session):
    if not os.path.exists(url.database):
        RegulASTable.metadata.create_all(engine)


class Data(RegulASTable):

    __tablename__ = 'Data'

    name = Column('name', String(256))
    meta = Column('meta', String(4096))
    num_samples = Column('num_samples', Integer)
    num_features = Column('num_features', Integer)
    md5 = Column('md5', String(32), unique=True)

    experiments = relationship('Experiment', back_populates='data')


class Pipeline(RegulASTable):

    __tablename__ = 'Pipeline'

    experiment_idx = Column('experiment_id', ForeignKey('Experiment.ID'))
    success = Column('success', Boolean, nullable=False)

    transformations = relationship('TransformationSequence', back_populates='pipeline')
    experiment = relationship('Experiment', back_populates='pipelines')


class Transformation(RegulASTable):

    class Type(enum.Enum):
        MODEL = enum.auto()
        TRANSFORM = enum.auto()

    __tablename__ = 'Transformation'

    fqn = Column('fqn', String(512), unique=True)
    version = Column('version', String(128))
    source = Column('source', String(65536))
    type_ = Column('type', Enum(Type))

    pipelines = relationship('TransformationSequence', back_populates='transformation')


class TransformationSequence(RegulASTable):

    __tablename__ = 'TransformationSequence'

    transformation_idx = Column('transformation_id', ForeignKey('Transformation.ID'))
    pipeline_idx = Column('pipeline_id', ForeignKey('Pipeline.ID'))
    position = Column('position', Integer, default=1)

    pipeline = relationship('Pipeline', back_populates='transformations')
    transformation = relationship('Transformation', back_populates='pipelines')
    hyper_parameters = relationship('HyperParameterValue', back_populates='transformation')


class HyperParameter(RegulASTable):

    __tablename__ = 'HyperParameter'

    name = Column('name', String(512), unique=True)

    transformations = relationship('HyperParameterValue', back_populates='hyper_parameter')


class HyperParameterValue(RegulASTable):

    __tablename__ = 'HyperParameterValue'

    transformation_idx = Column('transformation_id', ForeignKey('TransformationSequence.ID'))
    hyper_parameter_idx = Column('hyper_parameter_id', ForeignKey('HyperParameter.ID'))
    value = Column('value', String(128))

    transformation = relationship('TransformationSequence', back_populates='hyper_parameters')
    hyper_parameter = relationship('HyperParameter', back_populates='transformations')


class FeatureRanking(RegulASTable):

    __tablename__ = 'FeatureRanking'

    experiment_idx = Column('experiment_id', ForeignKey('Experiment.ID'))
    feature = Column('feature', String(128))
    score = Column('score', Float)

    experiment = relationship('Experiment', back_populates='feature_scores')


class Prediction(RegulASTable):

    __tablename__ = 'Prediction'

    experiment_idx = Column('experiment_id', ForeignKey('Experiment.ID'))
    sample_name = Column('sample_name', String(128))
    true_value = Column('true_value', Float)
    predicted_value = Column('predicted_value', Float)
    training = Column('training', Integer, nullable=False)
    fold = Column('fold', Integer)

    experiment = relationship('Experiment', back_populates='predictions')


class Experiment(RegulASTable):

    __tablename__ = 'Experiment'

    name = Column('name', String(1024))
    data_idx = Column('data_id', ForeignKey('Data.ID'))
    config = Column('config', String(65536))
    md5 = Column('md5', String(32), unique=True)
    random_seed = Column('random_seed', Integer)

    data = relationship('Data', back_populates='experiments')
    pipelines = relationship('Pipeline', back_populates='experiment')
    feature_scores = relationship('FeatureRanking', back_populates='experiment')
    predictions = relationship('Prediction', back_populates='experiment')
