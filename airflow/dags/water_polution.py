from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "Admin",
    "start_date": dt.datetime(2023, 11, 26),
    "retries": 5,
    "retry_delays": dt.timedelta(minutes=5),
    "depends_on_past": False
}

with DAG(
    dag_id='water_polution',
    default_args=args,
    schedule_interval=None,
    tags=['water_polution', 'score'],
) as dag:
    data_creation = BashOperator(task_id='data_creation',
    bash_command="python3 /home/antosha/project/scripts/data_creation.py",
    dag=dag)
    model_preprocessing = BashOperator(task_id='model_preprocessing',
    bash_command="python3 /home/antosha/project/scripts/model_preprocessing.py",
    dag=dag)
    model_preparation = BashOperator(task_id='model_preparation',
    bash_command="python3 /home/antosha/project/scripts/model_preparation.py",
    dag=dag)
    model_testing = BashOperator(task_id='model_testing',
    bash_command="python3 /home/antosha/project/scripts/model_testing.py",
    dag=dag)
    data_creation >> model_preprocessing >> model_preparation >> model_testing