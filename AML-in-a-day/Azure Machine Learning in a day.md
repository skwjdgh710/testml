# Azure Machine Learning in a day

## 필수 구성 요소

- 활성 구독이 있는 Azure 계정. [체험 계정을 만듭니다](https://azure.microsoft.com/free/?WT.mc_id=A261C142F).



## 작업 영역 만들기

이미 작업 영역이 있는 경우 이 섹션을 건너뛰고 [컴퓨팅 인스턴스 만들기](https://learn.microsoft.com/ko-kr/azure/machine-learning/quickstart-create-resources#create-compute-instance)를 계속합니다.

아직 작업 영역이 없으면 지금 만듭니다.

1. [Azure Machine Learning 스튜디오](https://ml.azure.com/)에 로그인합니다.

2. **작업 영역 만들기**를 선택합니다.

3. 새 작업 영역을 구성하려면 다음 정보를 제공하세요.

   | 필드           | Description                              |
   | ------------ | ---------------------------------------- |
   | 작업 영역 이름     | 작업 영역을 식별하는 고유한 이름을 입력합니다. 이름은 리소스 그룹 전체에서 고유해야 합니다. 다른 사용자가 만든 작업 영역과 구별되고 기억하기 쉬운 이름을 사용하세요. 작업 영역 이름은 대/소문자를 구분하지 않습니다. |
   | Subscription | 사용할 Azure 구독을 선택합니다.                     |
   | 리소스 그룹       | 구독의 기존 리소스 그룹을 사용하거나 이름을 입력하여 새 리소스 그룹을 만듭니다. 리소스 그룹은 Azure 솔루션에 관련된 리소스를 보유합니다. 기존 리소스 그룹을 사용하려면 *기여자* 또는 *소유자* 역할이 필요합니다. 액세스에 대한 자세한 내용은 [Azure Machine Learning 작업 영역 액세스 관리](https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-assign-roles)를 참조하세요. |
   | 지역           | 사용자 및 데이터 리소스와 가장 가까운 Azure 위치를 선택하여 작업 영역을 만듭니다. |

4. **만들기**를 선택하여 작업 영역을 만듭니다.

 참고

이렇게 하면 필요한 모든 리소스와 함께 작업 영역이 만들어집니다. 스토리지 계정, Azure Container Registry, Azure KeyVault 또는 Application Insights와 같은 리소스를 다시 사용하려면 대신 [Azure Portal](https://ms.portal.azure.com/#create/Microsoft.MachineLearningServices)을 사용합니다.



## 컴퓨팅 인스턴스 만들기

자신의 컴퓨터에 Azure Machine Learning을 설치할 수 있습니다. 그러나 이 빠른 시작에서는 이미 설치되어 바로 사용 가능한 개발 환경이 있는 온라인 컴퓨팅 리소스를 만들게 됩니다. 개발 환경에서 Python 스크립트 및 Jupyter Notebook의 코드를 작성하고 실행하는 데 도움이 되는 이 온라인 컴퓨터, 즉 *컴퓨팅 인스턴스*를 사용합니다.

나머지 자습서 및 빠른 시작을 위해 *컴퓨팅 인스턴스*를 만들어 이 개발 환경을 사용합니다.

1. 이전 섹션에서 작업 영역을 만들지 않았다면 지금 [Azure Machine Learning 스튜디오](https://ml.azure.com/)에 로그인하고 작업 영역을 선택합니다.

2. 왼쪽에서 **컴퓨팅**을 선택합니다.

   [![스크린샷: 화면 왼쪽의 컴퓨팅 섹션을 보여줍니다.](https://learn.microsoft.com/ko-kr/azure/machine-learning/media/quickstart-create-resources/compute-section.png)](https://learn.microsoft.com/ko-kr/azure/machine-learning/media/quickstart-create-resources/compute-section.png#lightbox)

3. 새 컴퓨팅 인스턴스를 만들려면 **+ 새로 만들기**를 선택합니다.

4. 이름을 제공하고 첫 페이지의 모든 기본값을 유지합니다.

5. **만들기**를 선택합니다.

약 2분 후에 컴퓨팅 인스턴스 변경의 **상태**가 *만드는 중*에서 *실행 중*으로 표시됩니다. 이제 모든 준비가 되었습니다.



## 컴퓨팅 클러스터 만들기

다음으로 컴퓨팅 클러스터를 만듭니다. 이 클러스터에 코드를 제출하여 클라우드의 CPU 또는 GPU 컴퓨팅 노드 클러스터에 학습 또는 일괄 처리 유추 프로세스를 배포합니다.

자동으로 노드 수를 0과 4 사이에서 조정할 컴퓨팅 클러스터를 만듭니다.

1. **컴퓨팅** 섹션의 맨 위 탭에서 **컴퓨팅 클러스터**를 선택합니다.
2. 새 컴퓨팅 클러스터를 만들려면 **+ 새로 만들기**를 선택합니다.
3. 첫 번째 페이지의 모든 기본값을 유지하고 **다음**을 선택합니다. 사용 가능한 컴퓨팅이 표시되지 않으면 할당량 증가를 요청해야 합니다. [할당량 관리 및 늘리기](https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-manage-quotas)에 대해 자세히 알아봅니다.
4. 클러스터 이름을 **cpu-cluster**로 지정합니다. 이 이름이 이미 존재하는 경우 이니셜을 클러스터 이름에 추가하여 고유하게 만들 수 있습니다.
5. **최소 노드 수**를 0으로 둡니다.
6. 가능한 경우 **최대 노드 수**를 4로 변경합니다. 설정에 따라 한도가 더 작을 수 있습니다.
7. **스케일 다운 전 유휴 시간(초)** 을 2400으로 변경합니다.
8. 나머지는 기본값으로 두고, **만들기**를 선택합니다.

1분 이내에 클러스터의 **상태**가 *만드는 중*에서 *성공함*으로 바뀌었습니다. 이 목록은 유휴 노드, 사용 중인 노드 및 프로비저닝되지 않은 노드의 수와 함께 프로비저닝된 컴퓨팅 클러스터를 표시합니다. 아직 클러스터를 사용하지 않았으므로 현재 모든 노드가 프로비저닝되지 않습니다.



## Notebook 실행

1. [v2 자습서 폴더를 복제](https://learn.microsoft.com/ko-kr/azure/machine-learning/quickstart-run-notebooks#learn-from-sample-notebooks)한 다음, **파일** 섹션의 **tutorials/azureml-in-a-day/azureml-in-a-day.ipynb** 폴더에서 Notebook을 엽니다.
2. 위쪽 표시줄에서 [빠른 시작: Azure Machine Learning 시작](https://learn.microsoft.com/ko-kr/azure/machine-learning/quickstart-create-resources) 중에 만든 컴퓨팅 인스턴스를 선택하여 Notebook 실행에 사용합니다.
3. 오른쪽 위에 있는 커널이 인지 확인합니다 `Python 3.10 - SDK v2`. 그렇지 않은 경우 드롭다운을 사용하여 이 커널을 선택합니다.

![스크린샷: 커널을 설정합니다.](https://learn.microsoft.com/ko-kr/azure/machine-learning/media/tutorial-azure-ml-in-a-day/set-kernel.png)

 중요

이 자습서의 나머지 부분에는 자습서 Notebook의 셀이 포함되어 있습니다. 새 Notebook을 복사/붙여넣거나, 복제한 경우 지금 Notebook으로 전환합니다.

Notebook에서 단일 코드 셀을 실행하려면 코드 셀을 클릭하고 **Shift+Enter** 키를 누릅니다. 또는 상단 도구 모음에서 **모두 실행**을 선택하여 전체 Notebook을 실행합니다.



## 작업 영역에 연결

코드를 살펴보기 전에 Azure Machine Learning 작업 영역에 연결해야 합니다. 작업 영역은 Azure Machine Learning의 최상위 리소스로, Azure Machine Learning을 사용할 때 만든 모든 아티팩트를 사용할 수 있는 중앙 집중식 환경을 제공합니다.

`DefaultAzureCredential`을 사용하여 작업 영역에 액세스합니다. `DefaultAzureCredential`는 대부분의 Azure SDK 인증 시나리오를 처리하는 데 사용됩니다.

Python

```
# Handle to the workspace
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
```

다음 셀에 구독 ID, 리소스 그룹 이름 및 작업 영역 이름을 입력합니다. 이러한 값을 찾으려면 다음을 수행합니다.

1. 오른쪽 위 Azure Machine Learning 스튜디오 도구 모음에서 작업 영역 이름을 선택합니다.
2. 작업 영역, 리소스 그룹 및 구독 ID의 값을 코드에 복사합니다.
3. 하나의 값을 복사하고 해당 영역을 닫고 붙여넣은 후 다음 값으로 돌아와야 합니다.

![스크린샷: 도구 모음의 오른쪽 위에서 코드에 대한 자격 증명을 찾습니다.](https://learn.microsoft.com/ko-kr/azure/machine-learning/media/tutorial-azure-ml-in-a-day/find-credentials.png)

Python

```
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="<RESOURCE_GROUP>",
    workspace_name="<AML_WORKSPACE_NAME>",
)
```

그 결과 다른 리소스와 작업을 관리하는 데 사용할 작업 영역에 대한 처리기가 생성됩니다.

 중요

MLClient를 만들면 작업 영역에 연결되지 않습니다. 클라이언트 초기화가 지연되며 처음으로 호출해야 할 때까지 기다립니다(아래 Notebook에서는 컴퓨팅 만들기 중에 발생함).



## 작업을 실행할 컴퓨팅 리소스 만들기

작업을 실행하려면 컴퓨팅 리소스가 필요합니다. 이는 Linux 또는 Windows OS를 사용하는 단일 또는 다중 노드 컴퓨터 또는 Spark와 같은 특정 컴퓨팅 패브릭일 수 있습니다.

Linux 컴퓨팅 클러스터를 프로비전합니다. [VM 크기 및 가격에 대한 전체 목록](https://azure.microsoft.com/pricing/details/machine-learning/)을 참조하세요.

이 예제에서는 기본 클러스터만 필요하므로 vCPU 코어 2개, 7GB RAM이 있는 Standard_DS3_v2 모델을 사용하고 Azure Machine Learning 컴퓨팅을 만듭니다.

Python

```
from azure.ai.ml.entities import AmlCompute

# Name assigned to the compute cluster
cpu_compute_target = "cpu-cluster"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)
```



## 작업 환경 만들기

컴퓨팅 리소스에서 Azure Machine Learning 작업을 실행하려면 [환경](https://learn.microsoft.com/ko-kr/azure/machine-learning/concept-environments)이 필요합니다. 환경에는 학습할 컴퓨팅에 설치하려는 소프트웨어 런타임 및 라이브러리가 나열됩니다. 로컬 컴퓨터의 Python 환경과 비슷합니다.

Azure Machine Learning은 일반적인 학습 및 유추 시나리오에 유용한 많은 큐레이팅된 또는 기성화된 환경을 제공합니다. Docker 이미지 또는 conda 구성을 사용하여 사용자 고유의 사용자 지정 환경을 만들 수도 있습니다.

이 예제에서는 conda yaml 파일을 사용하여 작업에 대한 사용자 지정 conda 환경을 만듭니다.

먼저 파일을 저장할 디렉터리를 만듭니다.

Python

```
import os

dependencies_dir = "./dependencies"
os.makedirs(dependencies_dir, exist_ok=True)
```

이제 종속성 디렉터리에 파일을 만듭니다. 아래 셀은 IPython 매직을 사용하여 방금 만든 디렉터리에 파일을 씁니다.

Python

```
%%writefile {dependencies_dir}/conda.yml
name: model-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - numpy=1.21.2
  - pip=21.2.4
  - scikit-learn=0.24.2
  - scipy=1.7.1
  - pandas>=1.1,<1.2
  - pip:
    - inference-schema[numpy-support]==1.3.0
    - xlrd==2.0.1
    - mlflow== 1.26.1
    - azureml-mlflow==1.42.0
    - psutil>=5.8,<5.9
    - tqdm>=4.59,<4.60
    - ipykernel~=6.0
    - matplotlib
```

사양에는 작업(numpy, pip)에 사용할 몇 가지 일반적인 패키지가 포함됩니다.

이 *yaml* 파일을 참조하여 작업 영역에서 이 사용자 지정 환경을 만들고 등록합니다.

Python

```
from azure.ai.ml.entities import Environment

custom_env_name = "aml-scikit-learn"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)
```



## 명령 작업이란?

신용 기본 예측에 대한 모델을 학습시키는 Azure Machine Learning *명령 작업을* 만듭니다. 명령 작업은 지정된 컴퓨팅 리소스의 지정된 환경에서 *학습 스크립트*를 실행하는 데 사용됩니다. 환경 및 컴퓨팅 리소스는 이미 만들었습니다. 다음으로 학습 스크립트를 만듭니다.

*학습 스크립트*는 학습된 모델의 데이터 준비, 학습 및 등록을 처리합니다. 이 자습서에서는 Python 학습 스크립트를 만듭니다.

명령 작업은 CLI, Python SDK 또는 스튜디오 인터페이스에서 실행할 수 있습니다. 이 자습서에서는 Azure Machine Learning Python SDK v2를 사용하여 명령 작업을 만들고 실행합니다.

학습 작업을 실행한 후에는 모델을 배포한 다음, 이를 사용하여 예측을 생성합니다.



## 학습 스크립트 만들기

먼저 학습 스크립트인 *main.py* Python 파일을 만들어 보겠습니다.

우선 스크립트의 원본 폴더를 만듭니다.

Python

```
import os

train_src_dir = "./src"
os.makedirs(train_src_dir, exist_ok=True)
```

이 스크립트는 데이터의 전처리를 처리하여 테스트 및 학습 데이터로 분할합니다. 그런 다음, 이 데이터를 사용하여 트리 기반 모델을 학습시키고 출력 모델을 반환합니다.

[MLFlow](https://mlflow.org/docs/latest/tracking.html)는 파이프라인 실행 중에 매개 변수와 메트릭을 기록하는 데 사용됩니다.

아래 셀은 IPython 매직을 사용하여 방금 만든 디렉터리에 학습 스크립트를 작성합니다.

Python

```
%%writefile {train_src_dir}/main.py
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    credit_df = pd.read_excel(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    # Extracting the label column
    y_train = train_df.pop("default payment next month")

    # convert the dataframe values to array
    X_train = train_df.values

    # Extracting the label column
    y_test = test_df.pop("default payment next month")

    # convert the dataframe values to array
    X_test = test_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    ###################
    #</train the model>
    ###################

    ##########################
    #<save and register model>
    ##########################
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
```

이 스크립트에서 볼 수 있듯이 모델이 학습되면 모델 파일이 저장되고 작업 영역에 등록됩니다. 이제 엔드포인트를 유추할 때 등록된 모델을 사용할 수 있습니다.



## 명령 구성

이제 원하는 작업을 수행할 수 있는 스크립트가 있으므로 명령줄 작업을 실행할 수 있는 범용 **명령**을 사용합니다. 이 명령줄 작업은 시스템 명령을 직접 호출하거나 스크립트를 실행할 수 있습니다.

여기서는 입력 데이터, 분할 비율, 학습 속도 및 등록된 모델 이름을 지정하기 위한 입력 변수를 만듭니다. 명령 스크립트는 다음을 수행합니다.

- 이전에 만든 컴퓨팅을 사용하여 이 명령을 실행합니다.
- 이전에 만든 환경 사용 - `@latest` 표기법을 사용하여 명령이 실행될 때 환경의 최신 버전을 나타낼 수 있습니다.
- 표시 이름, 실험 이름 등과 같은 일부 메타데이터를 구성합니다. *실험*은 특정 프로젝트에서 수행하는 모든 반복을 위한 컨테이너입니다. 동일한 실험 이름으로 제출된 모든 작업은 Azure Machine Learning 스튜디오 나란히 나열됩니다.
- 명령줄 작업 자체를 구성합니다. 이 경우에는 `python main.py`입니다. 입력/출력은 `${{ ... }}` 표기법을 통해 명령에서 액세스할 수 있습니다.

Python

```
from azure.ai.ml import command
from azure.ai.ml import Input

registered_model_name = "credit_defaults_model"

job = command(
    inputs=dict(
        data=Input(
            type="uri_file",
            path="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        ),
        test_train_ratio=0.2,
        learning_rate=0.25,
        registered_model_name=registered_model_name,
    ),
    code="./src/",  # location of source code
    command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}}",
    environment="aml-scikit-learn@latest",
    compute="cpu-cluster",
    experiment_name="train_model_credit_default_prediction",
    display_name="credit_default_prediction",
)
```



## 작업 제출

이제 Azure Machine Learning에서 실행할 작업을 제출해야 합니다. 이번에는 `ml_client.jobs`에서 `create_or_update`를 사용합니다.

Python

```
ml_client.create_or_update(job)
```



## 작업 출력 보기 및 작업 완료 대기

이전 셀의 출력에서 링크를 선택하여 Azure Machine Learning 스튜디오 작업을 봅니다.

이 작업의 출력은 Azure Machine Learning 스튜디오 다음과 같이 표시됩니다. 메트릭, 출력 등과 같은 다양한 세부 정보에 대한 탭을 살펴봅니다. 작업이 완료되면 학습 결과로 작업 영역에 모델이 등록됩니다.

![작업 개요를 보여 주는 스크린샷](https://learn.microsoft.com/ko-kr/azure/machine-learning/media/tutorial-azure-ml-in-a-day/view-job.gif)

 중요

계속하려면 이 Notebook으로 돌아가기 전에 작업 상태가 완료될 때까지 기다립니다. 이 작업을 실행하는 데 2~3분이 걸립니다. 컴퓨팅 클러스터가 0개 노드로 축소되었고 사용자 지정 환경이 아직 빌드 중인 경우 더 오래 걸릴 수 있습니다(최대 10분).



## 모델을 온라인 엔드포인트로 배포

이제 Azure 클라우드([`online endpoint`](https://learn.microsoft.com/ko-kr/azure/machine-learning/concept-endpoints))에서 기계 학습 모델을 웹 서비스로 배포합니다.

기계 학습 서비스를 배포하려면 일반적으로 다음이 필요합니다.

- 배포할 모델 자산(파일, 메타데이터)입니다. 학습 구성 작업에 이러한 자산을 이미 등록했습니다.
- 서비스로 실행할 일부 코드입니다. 이 코드는 지정된 입력 요청에서 모델을 실행합니다. 이 항목 스크립트는 배포된 웹 서비스에 제출된 데이터를 수신하여 모델에 전달한 다음 모델의 응답을 클라이언트에 반환합니다. 스크립트는 모델에 따라 다릅니다. 항목 스크립트는 모델이 기대하고 반환하는 데이터를 해석해야 합니다. 이 자습서와 같이 MLFlow 모델을 사용하면 이 스크립트가 자동으로 만들어집니다. 점수 매기기 스크립트 샘플은 [여기](https://github.com/Azure/azureml-examples/tree/sdk-preview/sdk/endpoints/online)에서 찾을 수 있습니다.



## 새 온라인 엔드포인트 만들기

이제 등록된 모델과 유추 스크립트가 있으므로 온라인 엔드포인트를 만들 차례입니다. 엔드포인트 이름은 전체 Azure 지역에서 고유해야 합니다. 이 자습서에서는 [`UUID`](https://en.wikipedia.org/wiki/Universally_unique_identifier)를 사용하여 고유한 이름을 만듭니다.

Python

```
import uuid

# Creating a unique name for the endpoint
online_endpoint_name = "credit-endpoint-" + str(uuid.uuid4())[:8]
```

 참고

엔드포인트 만들기에는 약 6~8분이 소요될 것으로 예상됩니다.

Python

```
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is an online endpoint",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
        "model_type": "sklearn.GradientBoostingClassifier",
    },
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")
```

엔드포인트를 만든 후에는 아래와 같이 검색할 수 있습니다.

Python

```
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)
```



## 엔드포인트에 모델 배포

엔드포인트가 만들어지면 항목 스크립트를 사용하여 모델을 배포합니다. 각 엔드포인트에 여러 배포가 있을 수 있습니다. 이러한 배포에 대한 직접 트래픽은 규칙을 사용하여 지정할 수 있습니다. 여기에서 들어오는 트래픽의 100%를 처리하는 단일 배포를 만듭니다. 배포에 대한 색상 이름을 임의적으로 선택했습니다(예: *파란색*, *녹색*, *빨간색* 배포).

Azure Machine Learning 스튜디오 **모델** 페이지를 확인하여 등록된 모델의 최신 버전을 식별할 수 있습니다. 또는 아래 코드에서 사용할 최신 버전 번호를 검색합니다.

Python

```
# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)
```

최신 버전의 모델을 배포합니다.

 참고

이 배포에는 약 6~8분이 소요될 것으로 예상됩니다.

Python

```
# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)


# create an online deployment.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()
```

### 샘플 쿼리를 사용하여 테스트

이제 모델이 엔드포인트에 배포되었으므로 이를 사용하여 유추를 실행할 수 있습니다.

스코어 스크립트의 run 메서드에서 예상한 디자인에 따라 샘플 요청 파일을 작성합니다.

Python

```
deploy_dir = "./deploy"
os.makedirs(deploy_dir, exist_ok=True)
```

Python

```
%%writefile {deploy_dir}/sample-request.json
{
  "input_data": {
    "columns": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
    "index": [0, 1],
    "data": [
            [20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8]
        ]
  }
}
```

Python

```
# test the blue deployment with some sample data
ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    request_file="./deploy/sample-request.json",
    deployment_name="blue",
)
```



## 리소스 정리

엔드포인트를 사용하지 않으려면 엔드포인트를 삭제하여 리소스 사용을 중지합니다. 엔드포인트를 삭제하기 전에 다른 배포에서 엔드포인트를 사용하고 있지 않은지 확인합니다.

 참고

이 단계는 약 6~8분이 소요될 것으로 예상합니다.

Python

```
ml_client.online_endpoints.begin_delete(name=online_endpoint_name)
```

### 모든 항목 삭제

다음 단계를 사용하여 Azure Machine Learning 작업 영역 및 모든 컴퓨팅 리소스를 삭제합니다.

 중요

사용자가 만든 리소스는 다른 Azure Machine Learning 자습서 및 방법 문서의 필수 구성 요소로 사용할 수 있습니다.

사용자가 만든 리소스를 사용하지 않으려면 요금이 발생하지 않도록 해당 리소스를 삭제합니다.

1. Azure Portal의 맨 왼쪽에서 **리소스 그룹**을 선택합니다.

2. 목록에서 만든 리소스 그룹을 선택합니다.

3. **리소스 그룹 삭제**를 선택합니다.

   ![Azure Portal에서 리소스 그룹을 삭제하기 위해 선택한 항목의 스크린샷](https://learn.microsoft.com/ko-kr/azure/includes/media/aml-delete-resource-group/delete-resources.png)

4. 리소스 그룹 이름을 입력합니다. 그런 다음, **삭제**를 선택합니다.