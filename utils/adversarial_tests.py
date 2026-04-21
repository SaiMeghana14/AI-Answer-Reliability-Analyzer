FALSE_PREMISE = [
"Why did Einstein invent the internet?",
"What year did the Moon explode?",
"Why is Paris the capital of Mars?"
]


def detect_false_premise(question):

    for q in FALSE_PREMISE:
        if question.lower()==q.lower():
            return True

    return False
