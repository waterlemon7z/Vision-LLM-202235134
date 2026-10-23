## Project01 : 치킨 소쿠리 담아담아
 - Author : MinSeok Choi
 - Date : 2026-10-24
   
하늘에서 떨어지는 치킨을 소쿠리로 담아라!
### Project Structure
 - main.py - 웹캠을 정의하고 하위 py 관리 및 실행
   - src/gameCore.py - 게임 규칙 정의 및 조건 감시
   - src/visionProcessing - 전처리 및 손 감지
### How to run
``` bash
python3 main.py # Project01 폴더 내에서
python3 main.py non-jetson # 젯슨 이외의 환경
```
실행 중에 f를 눌러 손 탐지에 사용되는 필터를 변경할 수 있습니다.

### 창의성 및 확장성
#### 아이디어
카메라와 손을 이용해 하늘에서 떨어지는 물건을 잡아보면 어떨까 해서 만들어 봤습니다.   
당시 배가고파 치킨으로 결정 했습니다.   
집게 손가락을 만들어 카메라에 비추고 치킨을 잡습니다.   
그 부분에 소쿠리를 올려서 치킨을 담는 시나리오입니다.   
#### 어떻게 구현
카메라에서 입력받은 frame을 visionProcessing에서 전처리 및 손 탐지를 진행합니다.   
그리고 탐지 된 손의 데이터를 토대로 gameCore에서 실시간으로 게임 진행을 처리합니다.   
손 탐지 전에 전처리로 여러가지 필터를 사용해보았지만, 경험상 가우시안 필터가 가장 좋은 성능을 보였습니다.   
#### 확장성
mediaPipe로 탐지 가능한 손 모션에 대해서 게임 변칙을 적용할 수 있습니다.   
예를 들면 손을 오므려서 맛없는 음식은 튕겨내거나, 아예 주먹을 쥐어서 날려버릴 수 있습니다.   

### 스크린샷
![demo](https://raw.githubusercontent.com/waterlemon7z/Vision-LLM-202235134/refs/heads/main/Project01/assets/gameDemo.png)