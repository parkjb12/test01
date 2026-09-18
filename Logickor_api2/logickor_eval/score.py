import argparse
import glob

import pandas as pd

# 파일 경로 패턴
# file_pattern = './judge_20240418_103542.jsonl'
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--print", help="judge Output File Location", default=None)
args = parser.parse_args()

if args.print is None:
    raise ValueError("Judge Output File Location is required")

# ⬇️추가한 코드
def extract_scores(item, file_path):
    # single_score = item["query_single"]["judge_score"]
    # multi_score = item["query_multi"]["judge_score"]
    # ⬇️추가한 코드
    if "query_single" in item and "query_multi" in item:
        return item["query_single"]["judge_score"], item["query_multi"]["judge_score"]

    if "judge_single_score" in item and "judge_multi_score" in item:
        return item["judge_single_score"], item["judge_multi_score"]

    raise ValueError(
        f"[Invalid input format] {file_path}\n"
        "This file does not contain judge fields.\n"
        "Expected keys: query_single/query_multi or judge_single_score/judge_multi_score.\n"
        "Use evaluator output JSONL (e.g., ./evaluated/.../*.jsonl), not raw generated output."
    )

# 카테고리별 점수 집계를 위한 딕셔너리
category_scores = {}

# 전체 싱글 점수와 멀티 점수의 리스트
total_single_scores = []
total_multi_scores = []

# 심판이 점수를 내지 못한 항목 수. evaluator.py 가 judge_score=null 로 기록하며,
# 이를 0 점으로 평균에 넣으면 전체 점수가 실제보다 크게 낮아지므로 평균에서 제외한다.
unscored_single = 0
unscored_multi = 0

# 지정된 패턴에 맞는 모든 파일을 찾아서 처리
# file_paths = glob.glob(args.print)
# ⬇️추가한 코드
file_paths = glob.glob(args.print, recursive=True)
# ⬇️추가한 코드
if not file_paths:
    raise ValueError(f"No files matched pattern: {args.print}")

# for file_path in glob.glob(args.print):
# ⬇️추가한 코드
for file_path in file_paths:
    file = pd.read_json(file_path, orient="records", encoding="utf-8-sig", lines=True)
    for item in file.to_dict(orient="records"):
        category = item["category"]
        # single_score = item["query_single"]["judge_score"]
        # multi_score = item["query_multi"]["judge_score"]
        # ⬇️추가한 코드
        single_score, multi_score = extract_scores(item, file_path)

        if category not in category_scores:
            category_scores[category] = {"single_scores": [], "multi_scores": []}

        # ⬇️추가한 코드: 채점 실패(null)는 0 점이 아니라 '측정 불가'이므로 평균에서 뺀다.
        if single_score is None:
            unscored_single += 1
        else:
            category_scores[category]["single_scores"].append(single_score)
            total_single_scores.append(single_score)

        if multi_score is None:
            unscored_multi += 1
        else:
            category_scores[category]["multi_scores"].append(multi_score)
            total_multi_scores.append(multi_score)

# 표의 헤더 생성
table_header = "| Category | Single turn | Multi turn |\n|---|---|---|"

# 표의 내용 생성
# ⬇️추가한 코드
def mean_or_none(values):
    return sum(values) / len(values) if values else None


# ⬇️추가한 코드
def fmt(value):
    return f"{value:.2f}" if value is not None else "N/A"


table_rows = []
for category, scores in category_scores.items():
    avg_single = mean_or_none(scores["single_scores"])
    avg_multi = mean_or_none(scores["multi_scores"])
    table_rows.append(f"| {category} | {fmt(avg_single)} | {fmt(avg_multi)} |")

    # total_single_scores.extend(scores["single_scores"])
    # total_multi_scores.extend(scores["multi_scores"])
    # ⬇️추가한 코드
    # 이미 파일 읽는 루프에서 누적했으므로 여기서 다시 더하지 않는다.

# 카테고리별 점수 평균 출력
print(table_header)
for row in table_rows:
    print(row)

# 전체 점수의 평균 계산 및 출력
avg_total_single = mean_or_none(total_single_scores)
avg_total_multi = mean_or_none(total_multi_scores)
combined = [v for v in (avg_total_single, avg_total_multi) if v is not None]
avg_total = sum(combined) / len(combined) if combined else None

# 전체 점수 평균 출력
print("\n| Category | Score |\n|---|---|")
print(f"| Single turn | {fmt(avg_total_single)} |")
print(f"| Multi turn | {fmt(avg_total_multi)} |")
print(f"| Overall | {fmt(avg_total)} |")

# ⬇️추가한 코드: 제외된 항목을 눈에 띄게 알린다. 이 값이 0 이 아니면 점수는 일부 표본에
# 대한 것이므로, 실패한 항목만 다시 채점해야 신뢰할 수 있다. 파일을 통째로 지우면 이미
# 성공한 채점까지 다시 하게 되므로 --rejudge-failed 로 실패 항목만 메우는 편이 낫다.
unscored_total = unscored_single + unscored_multi
graded_total = len(total_single_scores) + len(total_multi_scores)
if unscored_total:
    print(
        f"\n! 채점 실패로 평균에서 제외된 항목: {unscored_total}개"
        f" (single {unscored_single}, multi {unscored_multi}) / 채점 성공 {graded_total}개."
        "\n  위 점수는 부분 표본 기준입니다. 실패한 항목만 다시 채점하려면:"
        "\n    python logickor_eval/evaluator.py -o generated/<모델경로> -j <심판모델> --rejudge-failed"
    )
