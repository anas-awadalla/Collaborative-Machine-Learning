# cloud-project

Distributed machine learning over the internet.

Final project for [CSE 453](https://courses.cs.washington.edu/courses/cse453/21au/)

## Development

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

Listens on port `5000` by default, but can be overriden with an argument: `python app.py 3000`.



TODO: 

- [ ] Figure out python model optimizing (converting client gradients to python tensors) - Anas
- [ ] Data loading and splitting on client side - Vishal
- [ ] Load balancing
- [ ] Get a larger dataset/model working