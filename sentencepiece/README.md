# SentencePiece

[![Build C++](https://github.com/google/sentencepiece/actions/workflows/cmake.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/cmake.yml)
[![Build Wheels](https://github.com/google/sentencepiece/actions/workflows/wheel.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/wheel.yml)
[![GitHub Issues](https://img.shields.io/github/issues/google/sentencepiece.svg)](https://github.com/google/sentencepiece/issues)
[![PyPI version](https://badge.fury.io/py/sentencepiece.svg)](https://badge.fury.io/py/sentencepiece)
[![PyPi downloads](https://img.shields.io/pypi/dm/sentencepiece?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sentencepiece/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

SentencePiece는 어휘 크기가 신경망 모델 훈련 전에 미리 결정되는 주로 신경망 기반 텍스트 생성 시스템을 위한 비지도 텍스트 토크나이저 및 디토크나이저입니다. SentencePiece는 **서브워드 단위** (예: **바이트 페어 인코딩 (BPE)** [[Sennrich 등](https://www.aclweb.org/anthology/P16-1162)]) 및 **유니그램 언어 모델** [[Kudo.](https://arxiv.org/abs/1804.10959)])을 구현하며 원시 문장에서 직접 훈련하는 확장 기능을 제공합니다. SentencePiece를 사용하면 언어별 사전/사후 처리에 의존하지 않는 순수한 엔드 투 엔드 시스템을 만들 수 있습니다.

## 기술적 특징
- **Purely data driven(순수 데이터 기반)**: SentencePiece는 문장에서 토크나이징 및 디토크나이징 모델을 훈련합니다. 사전 토크나이징 ([Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)/[MeCab](http://taku910.github.io/mecab/)/[KyTea](http://www.phontron.com/kytea/))는 항상 필요하지 않습니다.
- **Language independent(언어 독립적)**: SentencePiece는 문장을 유니코드 문자의 시퀀스로만 처리합니다. 언어에 종속적인 로직이 없습니다.
- **Multiple subword algorithms(다양한 서브워드 알고리즘)**: **BPE** [[Sennrich 등](https://www.aclweb.org/anthology/P16-1162)] 및 **유니그램 언어 모델** [[Kudo.](https://arxiv.org/abs/1804.10959)]이 지원됩니다.
- **Subword regularization(서브워드 정규화)**: SentencePiece는 [서브워드 정규화](https://arxiv.org/abs/1804.10959) 및 [BPE-dropout](https://arxiv.org/abs/1910.13267)에 대한 서브워드 샘플링을 구현하여 NMT 모델의 견고성과 정확도를 향상시킵니다.
- **Fast and lightweight(빠르고 가벼움)**: 세그멘테이션 속도는 약 50k 문장/초이며 메모리 사용량은 약 6MB입니다.
- **Self-contained(자체 포함)**: 동일한 모델 파일을 사용하는 한 동일한 토크나이징/디토크나이징이 얻어집니다.
- **Direct vocabulary id generation(직접 어휘 ID 생성)**: SentencePiece는 어휘 ID 매핑을 관리하며 원시 문장에서 직접 어휘 ID 시퀀스를 생성할 수 있습니다.
- **NFKC-based normalization(NFKC 기반 정규화)**: SentencePiece는 NFKC 기반 텍스트 정규화를 수행합니다.

SentencePiece 소프트웨어/알고리즘에 익숙하지 않은 사람들은 [여기에서 간단한 소개를 읽을 수 있습니다](https://medium.com/@jacky2wong/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08).

## 다른 구현과의 비교
|기능|SentencePiece|[subword-nmt](https://github.com/rsennrich/subword-nmt)|[WordPiece](https://arxiv.org/pdf/1609.08144.pdf)|
|:---|:---:|:---:|:---:|
|지원되는 알고리즘|BPE, 유니그램, 문자, 단어|BPE|BPE*|
|OSS?|예|예|Google 내부|
|서브워드 정규화|[예](#subword-regularization-and-bpe-dropout)|아니오|아니오|
|Python 라이브러리 (pip)|[예](python/README.md)|아니오|N/A|
|C++ 라이브러리|[예](doc/api.md)|아니오|N/A|
|사전 세그멘테이션 필요?|[아니오](#whitespace-is-treated-as-a-basic-symbol)|예|예|
|사용자 정의 정규화 (예: NFKC)|[예](doc/normalization.md)|아니오|N/A|
|직접 ID 생성|[예](#end-to-end-example)|아니오|N/A|

WordPiece에서 사용되는 BPE 알고리즘은 원래의 BPE와 약간 다릅니다.

## 개요
### SentencePiece란 무엇인가요?
SentencePiece는 신경 기계 번역에서 열린 어휘 문제를 완화하는 효과적인 방법인 **서브-워드 단위**의 재구현입니다. SentencePiece는 두 가지 세그멘테이션 알고리즘, **바이트 페어 인코딩 (BPE)** [[Sennrich 등](http://www.aclweb.org/anthology/P16-1162)] 및 **유니그램 언어 모델** [[Kudo.](https://arxiv.org/abs/1804.10959)]을 지원합니다. 다른 구현과의 주요 차이점은 다음과 같습니다.

#### 고유 토큰 수는 미리 결정됩니다.
신경 기계 번역 모델은 일반적으로 고정된 어휘로 작동합니다. 대부분의 비지도 단어 세그멘테이션 알고리즘과 달리 무한 어휘를 가정하는 대신, SentencePiece는 최종 어휘 크기가 고정되도록 세그멘테이션 모델을 훈련합니다. 예를 들어, 8k, 16k 또는 32k입니다.

SentencePiece는 훈련을 위한 최종 어휘 크기를 지정하는데, 이는 병합 작업 수를 사용하는 [subword-nmt](https://github.com/rsennrich/subword-nmt)와 다릅니다. 병합 작업 수는 BPE 특정 매개변수이며 다른 세그멘테이션 알고리즘, 포함하여 유니그램, 단어 및 문자에는 적용되지 않습니다.

#### 원시 문장에서 훈련합니다.
이전 서브-워드 구현은 입력 문장이 사전 토크나이징되었다고 가정합니다. 이 제약은 효율적인 훈련을 위해 필요했지만, 언어 종속 토크나이저를 미리 실행해야 하므로 전처리가 복잡해집니다. SentencePiece의 구현은 모델을 원시 문장에서 빠르게 훈련할 수 있을 정도로 빠릅니다. 이것은 단어 사이에 명확한 공백이 없는 중국어와 일본어의 토크나이저와 디토크나이저를 훈련하는 데 유용합니다.

#### 공백은 기본 심볼로 취급됩니다.
자연어 처리의 첫 번째 단계는 텍스트 토크나이징입니다. 예를 들어, 표준 영어 토크나이저는 "Hello world." 텍스트를 다음 세 토큰으로 분할합니다.

> [Hello] [World] [.]

한 가지 관찰은 원래 입력과 토크나이징된 시퀀스가 **역으로 변환 가능하지 않다는 것**입니다. 예를 들어, "World"와 "." 사이에 공백이 없는 정보는 토크나이징된 시퀀스에서 삭제됩니다. 예를 들어, `Tokenize(“World.”) == Tokenize(“World .”)`

SentencePiece는 입력 텍스트를 유니코드 문자의 시퀀스로만 처리합니다. 공백도 일반 심볼로 처리됩니다. 공백을 명시적으로 기본 토큰으로 처리하기 위해 SentencePiece는 먼저 다음과 같이 공백을 메타 심볼 "▁" (U+2581)로 이스케이프합니다.

> Hello▁World.

그런 다음 이 텍스트는 예를 들면 다음과 같이 작은 조각으로 분할됩니다.

> [Hello] [▁Wor] [ld] [.]

세그멘티드 텍스트에 공백이 보존되므로 모호성 없이 텍스트를 디토크나이징할 수 있습니다.

```
  detokenized

 = ''.join(pieces).replace('▁', ' ')
```

이 기능을 사용하면 언어별 리소스에 의존하지 않고 디토크나이징을 수행할 수 있습니다.

표준 단어 세그멘터로 문장을 분할할 때 동일한 손실 없는 변환을 적용할 수 없습니다. 왜냐하면 그들은 공백을 특별한 심볼로 취급하기 때문입니다. 토크나이징된 시퀀스는 원래 문장을 복원하기 위해 필요한 정보를 보존하지 않습니다.

* (en) Hello world.   → [Hello] [World] [.]   \(Hello와 World 사이에 공백이 있음\)
* (ja) こんにちは世界。  → [こんにちは] [世界] [。] \(こんにちは와 世界 사이에 공백이 없음\)

#### 서브워드 정규화와 BPE-드롭아웃
서브워드 정규화 [[Kudo.](https://arxiv.org/abs/1804.10959)] 및 BPE-드롭아웃 [Provilkov 등](https://arxiv.org/abs/1910.13267)은 가상의 훈련 데이터를 온-더-플라이 서브워드 샘플링으로 강화하는 간단한 정규화 방법입니다. 이는 NMT 모델의 정확성뿐만 아니라 견고성을 향상시키는 데 도움이 됩니다.

서브워드 정규화를 활성화하려면 SentencePiece 라이브러리
([C++](doc/api.md#sampling-subword-regularization)/[Python](python/README.md))를 NMT 시스템에 통합하여 매개변수 업데이트마다 하나의 세그멘테이션을 샘플링해야 합니다. 이는 표준 오프라인 데이터 준비와 다릅니다. [Python 라이브러리](python/README.md)의 예제입니다. 'New York'이 ``SampleEncode (C++)`` 또는 ``encode with enable_sampling=True (Python)`` 호출마다 다르게 세그멘티드되는 것을 확인할 수 있습니다. 샘플링 매개변수의 세부 사항은 [sentencepiece_processor.h](src/sentencepiece_processor.h)에서 찾을 수 있습니다.

```
>>> import sentencepiece as spm
>>> s = spm.SentencePieceProcessor(model_file='spm.model')
>>> for n in range(5):
...     s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
...
['▁', 'N', 'e', 'w', '▁York']
['▁', 'New', '▁York']
['▁', 'New', '▁Y', 'o', 'r', 'k']
['▁', 'New', '▁York']
['▁', 'New', '▁York']
```

## 설치

### Python 모듈
SentencePiece는 SentencePiece 훈련 및 세그멘테이션을 지원하는 Python 래퍼를 제공합니다.
다음과 같이 SentencePiece의 Python 바이너리 패키지를 설치할 수 있습니다.

```
pip install sentencepiece
```

자세한 내용은 [Python 모듈](python/README.md)을 참조하세요.

### C++ 소스에서 SentencePiece 명령 줄 도구 빌드 및 설치
SentencePiece를 빌드하려면 다음 도구와 라이브러리가 필요합니다:

* [cmake](https://cmake.org/)
* C++11 컴파일러
* [gperftools](https://github.com/gperftools/gperftools) 라이브러리 (선택 사항, 10-40%의 성능 향상을 얻을 수 있습니다.)

Ubuntu에서는 apt-get으로 빌드 도구를 설치할 수 있습니다:
```
% sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
```

그런 다음 다음과 같이 명령 줄 도구를 빌드하고 설치할 수 있습니다.
```
% git clone https://github.com/google/sentencepiece.git 
% cd sentencepiece
% mkdir build
% cd build
% cmake ..
% make -j $(nproc)
% sudo make install
% sudo ldconfig -v
```
OSX/macOS에서는 마지막 명령을 `sudo update_dyld_shared_cache`로 바꿉니다.

### vcpkg를 사용하여 빌드 및 설치

[vcpkg](https://github.com/Microsoft/vcpkg) 종속성 관리자를 사용하여 sentencepiece를 다운로드하고 설치할 수 있습니다:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install sentencepiece

vcpkg의 sentencepiece 포트는 Microsoft 팀 멤버와 커뮤니티 기여자에 의해 최신 상태로 유지됩니다. 버전이 오래되었다면 [vcpkg 저장소](https://github.com/Microsoft/vcpkg)에서 문제를 생성하거나 풀 요청을 생성하세요.

### 서명된 릴리스 휠에서 SentencePiece 다운로드 및 설치

[GitHub 릴리스 페이지](https://github.com/google/sentencepiece/releases/latest)에서 휠을 다운로드할 수 있습니다.
릴리스 과정 중에 OpenSSF의 [slsa-framework/slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)를 사용하여 [SLSA3 서명](slsa.dev)을 생성합니다. 릴리스 바이너리를 검증하려면:
1. [sl

sa-framework/slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)를 사용하여 서명을 검증합니다.
2. [GitHub 릴리스 페이지](https://github.com/google/sentencepiece/releases/latest)에서 휠을 다운로드합니다.
3. pip를 사용하여 휠을 설치합니다.

## 사용법
SentencePiece는 다음 두 단계로 작동합니다.

1. **모델 훈련**: 원시 문장에서 SentencePiece 모델을 훈련합니다.
2. **세그멘테이션**: 훈련된 모델을 사용하여 문장을 세그멘테이션합니다.

### 모델 훈련
SentencePiece는 원시 문장에서 SentencePiece 모델을 훈련하는 명령 줄 도구 `spm_train`을 제공합니다. 다음은 `spm_train`의 기본 사용법입니다.

```
% spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --character_coverage=1.0 --model_type=unigram
```

- `--input`: 원시 문장 파일 (UTF-8 인코딩). 각 줄은 하나의 문장입니다.
- `--model_prefix`: 출력 모델 이름. `<model_name>.model` 및 `<model_name>.vocab` 두 개의 파일이 생성됩니다.
- `--vocab_size`: 어휘 크기. 어휘 크기는 모델 훈련 후에 결정됩니다.
- `--character_coverage`: 입력 문장의 문자를 모델링하는 데 사용되는 문자의 비율. 0.98의 경우 98%의 문자를 모델링하는 데 사용하고 나머지 2%는 `<unk>`로 처리됩니다. 일본어/한국어의 경우 0.9995, 나머지의 경우 1.0을 사용하는 것이 좋습니다.
- `--model_type`: 모델 유형. `unigram` (기본값), `bpe`, `char`, `word` 중 하나를 선택합니다.

`spm_train`의 모든 옵션은 `spm_train --help`로 확인할 수 있습니다.

### 세그멘테이션
SentencePiece는 훈련된 모델을 사용하여 문장을 세그멘테이션하는 명령 줄 도구 `spm_encode`를 제공합니다. 다음은 `spm_encode`의 기본 사용법입니다.

```
% echo "This is a test." | spm_encode --model=<model_name>.model
▁This ▁is ▁a ▁test .
```

- `--model`: 훈련된 모델 파일.

`spm_encode`의 모든 옵션은 `spm_encode --help`로 확인할 수 있습니다.

### Python/C++ API
SentencePiece는 Python 및 C++ API를 제공합니다. 자세한 내용은 [Python 모듈](python/README.md) 및 [C++ API](doc/api.md) 문서를 참조하세요.

## FAQ
### SentencePiece는 어떻게 작동하나요?
SentencePiece는 원시 문장에서 토크나이징 및 디토크나이징 모델을 훈련합니다. 토크나이징 및 디토크나이징은 동적 프로그래밍을 사용하여 최적의 세그멘테이션을 찾습니다. SentencePiece는 원시 문장을 유니코드 문자의 시퀀스로만 처리하며, 언어에 종속적인 로직이 없습니다. SentencePiece는 바이트 페어 인코딩 (BPE) 및 유니그램 언어 모델을 지원하며, 원시 문장에서 직접 훈련하는 확장 기능을 제공합니다.

### SentencePiece는 어떤 언어를 지원하나요?
SentencePiece는 언어에 종속적인 로직이 없으므로 모든 언어를 지원합니다. SentencePiece는 문장을 유니코드 문자의 시퀀스로만 처리합니다.

### SentencePiece의 성능은 어떻게 되나요?
SentencePiece의 세그멘테이션 속도는 약 50k 문장/초이며 메모리 사용량은 약 6MB입니다.

### SentencePiece는 어떤 알고리즘을 지원하나요?
SentencePiece는 바이트 페어 인코딩 (BPE) 및 유니그램 언어 모델을 지원합니다.

### SentencePiece는 어떻게 설치하나요?
SentencePiece는 Python 및 C++ API를 제공합니다. Python 바이너리 패키지는 `pip install sentencepiece`로 설치할 수 있습니다. C++ 소스에서 SentencePiece 명령 줄 도구를 빌드하려면 cmake, C++11 컴파일러 및 gperftools 라이브러리가 필요합니다.

### SentencePiece는 어떻게 사용하나요?
SentencePiece는 원시 문장에서 SentencePiece 모델을 훈련하는 명령 줄 도구 `spm_train`을 제공합니다. 훈련된 모델을 사용하여 문장을 세그멘테이션하는 명령 줄 도구 `spm_encode`도 제공됩니다. 또한 Python 및 C++ API를 통해 프로그래밍 방식으로 SentencePiece를 사용할 수 있습니다.

### SentencePiece의 주요 특징은 무엇인가요?
SentencePiece의 주요 특징은 순수 데이터 기반, 언어 독립적, 다양한 서브워드 알고리즘 지원, 서브워드 정규화, 빠르고 가벼운 세그멘테이션, 자체 포함, 직접 어휘 ID 생성, NFKC 기반 정규화 등입니다.

### reference
[SentencePiece_GitHub](https://github.com/google/sentencepiece)