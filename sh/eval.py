# pred_file_path="/ossfs/workspace/nas/agent/SWE/out5/preds.json"

import argparse
import json

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--pred_file_path', type=str, required=True, help='Path to the predictions file')
args=arg_parser.parse_args()
# {
#     "oauthlib__oauthlib.1fd52536.combine_file__ffdrn2le": {
#         "reward": 0,
#         "test_out":
#         "p2p": {

#We cal culate acc based on reward
record={}
results = {}
with open(args.pred_file_path, 'r') as f:
    results = json.load(f)
total = 500#len(results)
correct = 0
for key, value in results.items():
    #sympy__sympy-19954
    repo = key.split('-')[0]
    if value['reward'] > 0:
        correct += 1
    if repo not in record:
        record[repo] = [{key: value['reward']}]
    else:
        record[repo].append({key: value['reward']})
accuracy = correct / total if total > 0 else 0
print(f'Accuracy: {accuracy*100:.2f}% ({correct}/{total})')

acc_record = {}
# calculate accuracy per repo
for repo, entries in record.items():
    repo_total = len(entries)
    repo_correct = sum(1 for entry in entries if list(entry.values())[0] > 0)
    repo_accuracy = repo_correct / repo_total if repo_total > 0 else 0
    acc_record[repo] = {
        'accuracy': repo_accuracy,
        'correct': repo_correct,
        'total': repo_total
    }
    print(f'Repo: {repo}, Accuracy: {repo_accuracy*100:.2f}% ({repo_correct}/{repo_total})')
acc_record['total']={
    'accuracy': accuracy,
    'correct': correct,
    'total': total
}
# save the accuracy to a file under the same directory
import os
output_dir = os.path.dirname(args.pred_file_path)
output_file = os.path.join(output_dir, 'accuracy.json')
with open(output_file, 'w') as f:
    json.dump(acc_record, f, indent=4)
#             "reward": 1,