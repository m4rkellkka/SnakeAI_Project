import torch
import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No checkpoint path provided'}))
        return

    path = sys.argv[1]
    if not os.path.exists(path):
        print(json.dumps({'error': f'File not found: {path}'}))
        return

    try:
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        history = {
            'n_games': checkpoint.get('n_games', 0),
            'total_steps': checkpoint.get('total_steps', 0),
            'eval_games_history': checkpoint.get('eval_games_history', []),
            'eval_avg_history': checkpoint.get('eval_avg_history', []),
            'eval_max_history': checkpoint.get('eval_max_history', []),
            'loss_history': checkpoint.get('loss_history', []),
            'mean_loss_history': checkpoint.get('mean_loss_history', []),
            'score_history': checkpoint.get('score_history', []),
            'creation_date': os.path.getmtime(path)
        }
        print(json.dumps(history))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    main()
