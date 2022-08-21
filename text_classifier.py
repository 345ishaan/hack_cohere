import cohere
import load_examples

_API_KEY = 'pRNIiUHLK5vdWkiZvjTcYWJhMrPPBWTOAdf0dUND'

class Cohere:
    def __init__(self, api_key=_API_KEY):
        self.co = cohere.Client(f'{api_key}', '2021-11-08')
        self.examples = []
    
    def fill_examples(self):
        for e in load_examples.examples('doc_data', use_cache=True):
            self.examples.append(cohere.classify.Example(
                text=e[0], label=e[1]
            ))
    
    def classify(self, inputs):
        return self.co.classify(
            model='medium',
            inputs=inputs,
            examples=self.examples
        ).classifications

