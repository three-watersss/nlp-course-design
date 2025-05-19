def extract_lines_skip_first(filepath, ranges):
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for start, end in ranges:
        selected = [''.join(line.strip().split()[1:]) for line in lines[start-1:end]]
        results.extend(selected)

    return results


'''SVM方法与朴素贝叶斯方法'''
from spa.test import test_waimai
test_waimai()


'''情感词典法'''
file_path = 'spa/f_corpus/ch_waimai_corpus.txt'
line_ranges = [(3001, 4000), (7001, 8000)]
processed_lines = extract_lines_skip_first(file_path, line_ranges)

from spa.classifiers import DictClassifier

d = DictClassifier()
neg_correct = 0
pos_correct = 0

for i in range(len(processed_lines)):
    line = processed_lines[i]
    #print(line)
    result = d.analyse_sentence(line)
    if i >= 1000 and result == 1:
        pos_correct += 1
    elif i < 1000 and result == 0:
        neg_correct += 1

print('情感词典法', '正面语料正确数:', pos_correct)
print('情感词典法', '负面语料正确数:', neg_correct)


'''深度学习方法'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 载入 tokenizer 和模型
model_name = "uer/roberta-base-finetuned-dianping-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

neg_correct = 0
pos_correct = 0

for i in range(len(processed_lines)):
    line = processed_lines[i]
    #print(line)
    inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = [0, 1]
    predicted = labels[probs.argmax()]
    if i >= 1000 and predicted == 1:
        pos_correct += 1
    elif i < 1000 and predicted == 0:
        neg_correct += 1

print('Roberta:', '正面语料正确数:', pos_correct)
print('Roberta:', '负面语料正确数:', neg_correct)





