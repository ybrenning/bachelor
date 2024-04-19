## Error: RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

Occurs with torch 1.9.0

Solution: Downgrade to torch 1.8.1

```bash
pip install git+https://github.com/webis-de/small-text.git#egg=small-text[transformers] -I
pip install torch==1.8.1
pip install torchtext==0.9.1
```