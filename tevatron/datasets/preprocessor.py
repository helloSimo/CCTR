class TrainPreProcessor:
    def __init__(self, separator=' '):
        self.separator = separator

    def __call__(self, example):
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(text)

        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(text)

        return {'query': example['query'], 'positives': positives, 'negatives': negatives}


class QueryPreProcessor:
    def __init__(self):
        pass

    def __call__(self, example):
        return {'text_id': example['query_id'], 'text': example['query']}


class CorpusPreProcessor:
    def __init__(self, separator=' '):
        self.separator = separator

    def __call__(self, example):
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        return {'text_id': example['docid'], 'text': text}
