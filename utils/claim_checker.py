import re


def split_claims(text):
    claims = re.split(r'[.!?]+', text)
    return [c.strip() for c in claims if c.strip()]


def label_claim(claim, reference):

    words=set(claim.lower().split())
    ref=set(reference.lower().split())

    overlap=len(words.intersection(ref))

    if overlap > 6:
        return "Supported ✅"

    if overlap > 2:
        return "Unsupported ⚠️"

    return "Contradicted ❌"


def check_claims(answer, reference):

    claims=split_claims(answer)

    results=[]

    for c in claims:
        results.append(
            (c, label_claim(c, reference))
        )

    return results
