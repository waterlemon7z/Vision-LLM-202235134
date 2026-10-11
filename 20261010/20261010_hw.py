import numpy as np

score_str = "0.8825"

# TODO: score_str를 백분율로 표현한 뒤 정수로 변환하여 출력
score_int = int(float(score_str) * 100)
print(score_int)  # 예상 출력 "최종 점수: 88점"

scores = [[45, 88, 92],
          [30, 75, 60],
          [95, 40, 50]]

# TODO: List Comprehension을 활용해 50점 이상인 값만 통과시켜 담으세요.
passed_scores = [score for row in scores for score in row if score >= 50]
print("합격 점수들:", passed_scores)

# TODO: enumerate와 zip을 조합하거나 활용하여 80점 이상 학생 정보 출력
# TODO: 예상 출력 "1번 짱구 학생 (점수: 85점)"
students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]
for idx, (student, score) in enumerate(zip(students, scores), start=1):
    if score >= 80:
        print(f'{idx}번 학생 {student} | 점수 {score}')

scores = np.array([85, 90, 100, 60, 90, 75, 90])

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화
std = np.std(scores)
mean = np.mean(scores)

np_arr_normalized = (scores - mean) / std #perform normalize
print(np_arr_normalized)


# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0
scores = np.array([85, 90, 100, 60, 90, 75, 90], dtype=np.float32)
np_arr_final = (scores - scores.min()) / (scores.max() - scores.min()) #MinMax normalize
print(np_arr_final)
print(np_arr_final.dtype)

raw_students = [
    "zg0505_짱구",
    "cs0822_철수",
    "mg0910_맹구",
    "yr0605_유리",
    "hi0205_훈이"
    ]

raw_scores = np.array([
    [95, 82, 88],   # 짱구
    [70, 65, 90],   # 철수
    [98, 100, 92],  # 맹구
    [60, 75, 80],   # 유리
    [81, 88, 84],   # 훈이
])

subjects = ["국어", "수학", "과학"]

# TODO: list comprehension을 활용해 raw_students 리스트에서 이름만 추출한 리스트 생성
student_names = [raw_name.split('_')[1] for raw_name in raw_students]
print("1. 학생 명단: ", end="")
for idx, name in enumerate(student_names):
    sep = ", " if idx < len(student_names) - 1 else "\n\n"
    print(name, end=sep)


# TODO: 전산 시스템에 입력된 모든 성적들을 1차원 배열로 출력 (총 15개)
all_scores = raw_scores.flatten()
print("2. 전산 시스템에 입력된 모든 성적: ", end="")
for idx, score in enumerate(all_scores):
    sep = ", " if idx < len(all_scores) - 1 else "\n\n"
    print(f"{score}점", end=sep)

# TODO: 전체 12개 성적 중 85점 이상 개수
over_85 = np.bincount(all_scores>=85)[1]
print(f"3. 전체 15개 성적 중 85점 이상 개수: {over_85}개", end="\n\n")

# TODO: 학생별 평균 점수 출력
student_means = raw_scores.mean(axis=1)
print('4. 학생별 평균 점수, ', end='')
for name, mean in zip(student_names, student_means):
    print(f'{name}, {mean:.2f}점', end=', ')  # 예상 출력: "4. 학생별 평균 점수: 짱구 00점, 철수 00점, 맹구 00점, 유리 00점, 훈이 00점"
print()
# TODO: 평균 점수가 가장 높은 학생 선택
top_student_idx = student_means.argmax()
top_student_name = student_names[top_student_idx]
print(f'5. 최고점 학생 {top_student_name}, {student_means[top_student_idx]:.2f}')  # 예상 출력: "5. 최고 평균 수석 학생: XX (xx.xx점)"

# TODO: 전체 점수 기준, 최고 점수와 과목 그리고 해당 학생 계산 (modulo % 사용 추천)
overall_max_score = all_scores.max()
overall_max_idx = all_scores.argmax()
overall_max_subject = subjects[overall_max_idx % 3]
overall_max_student = student_names[overall_max_idx // 3]
print(f'6. 단일 과목 최고점 : {overall_max_score} ({overall_max_subject},{overall_max_student})')  # 예상 출력: "6. 단일 과목 최고 점수: xx점 (과목: XX, 학생: XX)"
