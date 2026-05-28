COT_PROMPT="""\

Context:
{context}

Question: {question}

Instruction: Let's think step by step. 
1. First, identify the relevant visual elements in the image.
2. Second, identify the most relevant sentences in the provided context.
3. Third, connect the visual and textual evidence to reason about the answer.
4. Finally, write a concise final answer on a new line that starts exactly with "Final Answer:". Keep it as short as possible.

Reasoning:"""

INFOSEEK_FEW_SHOTS = (
    {
        'context': [
            "The myrtle warbler (Setophaga coronata) is a small New World warbler.The myrtle warbler has a northerly and easterly distribution, with Audubon\'s further west.",
            "The summer male myrtle warbler has a slate blue back, and yellow crown, rump and flank patch. It has white tail patches, and the breast is streaked black.",
            # "The stem is greyish green, usually 7-20 (cm) (in Southern Europe exceptionally 40\xa0cm) high, unbranched, erect, and terete. Especially further up, the stem is covered with short transparent glandular hairs."
        ],
        'question': "What is the closest parent taxonomy of this bird?",
        'answer': "Myrtle Warbler"
    },
    {
        'context': [
            "When threatened, however, they can reach up to 40 (km/h) and maintain that speed for 150 (m). Wombats defend home territories centred on their burrows, and they react aggressively to intruders. The common wombat occupies a range of up to 23 (ha), while the hairy-nosed species have much smaller ranges, of no more than 4 (ha).",
            "Female wombats give birth to a single young after a gestation period of roughly 20–30 days, which varies between species. All species have well-developed pouches, which the young leave after about six to seven months. Wombats are weaned after 15 months, and are sexually mature at 18 months",
            # "For instance, the American Ornithologists' Union places larks just after the crows, shrikes, and vireos. At a finer level of detail, some now place the larks at the beginning of a superfamily Sylvioidea with the swallows, various \"Old World warbler\" and \"babbler\" groups, and others."
        ],
        'question': "How many offspring can this animal produce at the same time?",
        'answer': "1.0"
    },
    {
        'context': [
            # "In the summer of 2005, original owner Madeleine Paulson provided the Old Friends Equine retirement facility in Georgetown, Kentucky with a substantial gift to enable them to acquire the horse, bring him home from Japan, and look after him during his retirement years.On November 7, 2005, the seventeen-year-old Fraise died suddenly from a ruptured blood vessel in his abdomen.",
            "Fraise (1988–2005) was an American Thoroughbred racehorse best known for winning the 1992 Breeders' Cup Turf.## Background.Fraise was a bay horse bred by Allen E. Paulson.",
            "These are sometimes referred to as chip art, silicon art, silicon graffiti or silicon doodling. The 555 timer IC The Operational amplifier 7400-series integrated circuits 4000-series integrated circuits, the CMOS counterpart to the 7400 series (see also: 74HC00 series)"
        ],
        "question": "In which year was this animal born?",
        'answer': "1988"
    }
)

INFOSEEK_FEW_SHOTS_ELICITED = (
    {
        'context': [
            "<START_IMPORTANT_TEXT> The myrtle warbler (Setophaga coronata) is a small New World warbler <END_IMPORTANT_TEXT>.The myrtle warbler has a northerly and easterly distribution, with Audubon\'s further west.",
            "The summer male myrtle warbler has a slate blue back, and yellow crown, rump and flank patch. It has white tail patches, and the breast is streaked black.",
            # "The stem is greyish green, usually 7-20 (cm) (in Southern Europe exceptionally 40\xa0cm) high, unbranched, erect, and terete. Especially further up, the stem is covered with short transparent glandular hairs."
        ],
        'question': "What is the closest parent taxonomy of this bird?",
        'answer': "Myrtle Warbler"
    },
    {
        'context': [
            "When threatened, however, they can reach up to 40 (km/h) and maintain that speed for 150 (m). Wombats defend home territories centred on their burrows, and they react aggressively to intruders. The common wombat occupies a range of up to 23 (ha), while the hairy-nosed species have much smaller ranges, of no more than 4 (ha).",
            "<START_IMPORTANT_TEXT> Female wombats give birth to a single young after a gestation period of roughly 20–30 days, which varies between species. All species have well-developed pouches, which the young leave after about six to seven months <END_IMPORTANT_TEXT>. Wombats are weaned after 15 months, and are sexually mature at 18 months",
            # "For instance, the American Ornithologists' Union places larks just after the crows, shrikes, and vireos. At a finer level of detail, some now place the larks at the beginning of a superfamily Sylvioidea with the swallows, various \"Old World warbler\" and \"babbler\" groups, and others."
        ],
        'question': "How many offspring can this animal produce at the same time?",
        'answer': "1.0"
    },
    {
        'context': [
            # "In the summer of 2005, original owner Madeleine Paulson provided the Old Friends Equine retirement facility in Georgetown, Kentucky with a substantial gift to enable them to acquire the horse, bring him home from Japan, and look after him during his retirement years.On November 7, 2005, the seventeen-year-old Fraise died suddenly from a ruptured blood vessel in his abdomen.",
            "<START_IMPORTANT_TEXT> Fraise (1988–2005) was an American Thoroughbred racehorse best known for winning the 1992 Breeders' Cup Turf <END_IMPORTANT_TEXT>.## Background.Fraise was a bay horse bred by Allen E. Paulson. <END_IMPORTANT_TEXT>",
            "These are sometimes referred to as chip art, silicon art, silicon graffiti or silicon doodling. The 555 timer IC The Operational amplifier 7400-series integrated circuits 4000-series integrated circuits, the CMOS counterpart to the 7400 series (see also: 74HC00 series)"
        ],
        "question": "In which year was this animal born?",
        'answer': "1988"
    }
)

OVEN_FEW_SHOTS = (
    {
        'context': [
            "...",
            # "The summer male myrtle warbler has a slate blue back, and yellow crown, rump and flank patch. It has white tail patches, and the breast is streaked black.",
            # "The stem is greyish green, usually 7-20 (cm) (in Southern Europe exceptionally 40\xa0cm) high, unbranched, erect, and terete. Especially further up, the stem is covered with short transparent glandular hairs."
        ],
        'question': "what vegetable is on the left of the pamphlet?",
        'answer': "Celery"
    },
    {
        'context': [
            "...",
            # "Female wombats give birth to a single young after a gestation period of roughly 20–30 days, which varies between species. All species have well-developed pouches, which the young leave after about six to seven months. Wombats are weaned after 15 months, and are sexually mature at 18 months",
            # "For instance, the American Ornithologists' Union places larks just after the crows, shrikes, and vireos. At a finer level of detail, some now place the larks at the beginning of a superfamily Sylvioidea with the swallows, various \"Old World warbler\" and \"babbler\" groups, and others."
        ],
        'question': "what species of bear is this?",
        'answer': "American black bear"
    },
    {
        'context': [
            "..."
            # "In the summer of 2005, original owner Madeleine Paulson provided the Old Friends Equine retirement facility in Georgetown, Kentucky with a substantial gift to enable them to acquire the horse, bring him home from Japan, and look after him during his retirement years.On November 7, 2005, the seventeen-year-old Fraise died suddenly from a ruptured blood vessel in his abdomen.",
            # "These are sometimes referred to as chip art, silicon art, silicon graffiti or silicon doodling. The 555 timer IC The Operational amplifier 7400-series integrated circuits 4000-series integrated circuits, the CMOS counterpart to the 7400 series (see also: 74HC00 series)"
        ],
        "question": "what is the cat sitting on?",
        'answer': "Lap"
    }
)


VQA_PROMPT = (
"Answer the question based on the image above.\n\n"
"Question: {question}\n\n"
)

CONTEXT_SEP = '\n##\n\n'
CONTEXT_VQA_PROMPT = """
Given the context retrieved, answer the question based on the image above.
Question: {question}\n\n
Context:\n\n{context}\n\n
If the context does not help with the question, try to shortly answer it anyway.
"""
CONTEXT_VQA_PROMPT_training="""\
{question}

The following paragraphs may contain useful information to help answer the question correctly:

{context}
"""

CONTEXT_VQA_PROMPT_SELF_ELICIT = """\
Directly answer the question based on the context passages, no explanation is needed. 
If the context does not contain any evidence, output “I cannot answer based on the given context."

Question: {question}

Context:
{context}
"""

SYSTEM_PROMPT = """\
Answer the encyclopedic question about the given image. Don’t mention the visual content of image in your output. Directly output the answer of the question according to the context.
If the paragraphs do not contain the information required to answer the question, you should answer the question using your knowledge.\
"""

SELF_ELICIT_SYSTEM_PROMPT_VQA = "\n<START_IMPORTANT> and <END_IMPORTANT> are used to mark the important evidence you should focus on. Do not output the markers." # Poi tolto "you should focus on"

SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG = "\n<START_IMPORTANT_IMG> and <END_IMPORTANT_IMG> are used to mark the important visual evidence. Do not output the markers."
SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT = "\n<START_IMPORTANT_TEXT> and <END_IMPORTANT_TEXT> are used to mark the important textual evidence. Do not output the markers."

SELF_ELICIT_IMAGE_BBOX_SYSTEM_PROMPT_VQA = "\nFocus on the {entity} in the image highlighted by the {color} bbox."

PROMPT_SELF_ELICT_IMAGE_CROP = """\
This following image is a crop of the first image at coordinates "bbox_2d": {bbox_2d}, which contains the {entity} detected. Use this cropped image to help answer the question about the first image.\
"""

RELEVANCY_EVAL_SYSTEM_PROMPT = """\
You are a multimodal reasoning assistant specialized in Knowledge-Based Visual Question Answering (KB-VQA).
Your task is to evaluate whether a given text passage provides useful and relevant information for answering a question about an image.

You will be given:
- Image: a visual scene containing entities, actions, and context.
- Question: a natural-language question that refers to the image.
- Text Passage: an external knowledge snippet retrieved from a database or the web.

You must analyze the semantic alignment between the text, the image, and the question.
Follow these steps carefully before giving your final decision:
1. Understand the visual scene: Identify the key objects, people, actions, and context visible in the image.
2. Interpret the question: Determine what information the question seeks (e.g., factual, reasoning, counting, attribute-based).
3. Analyze the text passage: Extract the main claims, facts, and entities mentioned in the text.

Compare for relevance: Assess whether the information in the text:
- Contains at least one sentence that supports answering the question about the image, OR
- Provides background knowledge needed to interpret or reason about the image-question pair.

Important:
- If even a single sentence in the passage is relevant or useful, consider the entire passage as **relevant** and answer “Yes”.
- If no part of the passage contributes meaningfully to answering the question, answer “No”.

Output only one word:
"Yes" -> if the text provides relevant or useful information for answering the question.
"No" -> if the text is irrelevant or unhelpful.\
"""


# Variant: the retrieved passage ALWAYS includes a passage-image (an image that accompanies the text).
# Use this prompt when the passage is guaranteed to contain an image (for example a figure, diagram, or photo)
# and you want the model to evaluate the joint usefulness of the passage text + passage-image together for answering
# the question about the query image.
RELEVANCY_EVAL_SYSTEM_PROMPT_PASSAGE_WITH_IMAGE = """\
You are a multimodal reasoning assistant specialized in Knowledge-Based Visual Question Answering (KB-VQA).
The retrieved passage ALWAYS includes an accompanying image (the passage-image) together with text and an optional caption/alt-text.
Your task is to evaluate whether the passage (text + passage-image) provides useful and relevant information for answering a question about a separate query image.

You will be given:
- Query Image: the visual scene the question refers to.
- Question: a natural-language question about the query image.
- Passage: a text snippet accompanied by a passage-image. The passage-image should be treated as part of the evidence.
- If the passage-image is missing or not available, you will be given a simple black image that should be ignored.

You must analyze semantic and visual alignment across three sources: the query image, the question, and the passage (both its text and its passage-image).
Follow these steps before giving your final decision:
1. Inspect the query image: identify key objects, people, actions, attributes, and spatial/temporal context.
2. Parse the question: determine what information is required (factual, attribute, counting, reasoning, temporal, causal, etc.).
3. Read the passage text: extract claims, facts, entities, and any explicit answers or background knowledge.
4. Inspect the passage-image: identify visual elements, attributes, and relationships; evaluate whether it depicts evidence relevant to the question or to the query image.
5. Cross-check: determine whether the passage-image and/or passage text (alone or together) provide:
    - direct visual evidence that helps answer the question about the query image, OR
    - a textual sentence that directly supports answering the question, OR
    - background knowledge (text or image) necessary to interpret or reason about the query image-question pair.

Decision rule:
- If at least one sentence in the passage text OR the passage-image contains information that meaningfully helps answer the question about the query image, answer exactly and only "Yes".
- Otherwise, answer exactly and only "No".

Important formatting:
- Output a single token, either the capitalized word Yes or No, and nothing else.
"""

SECTION_EVAL_USER_TEMPLATE = """\
Here is the question on the image above: 
{question}

Here is the text passage to analyze:
{passage}

Does the text passage contain at least one sentence that may have some information usefule to answer the user question? 
"Yes"/"No" answer: \
"""


SYSTEM_PROMPT_REASONING = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>reasoning process here</think><answer>short answer here</answer>"


REASONING_TRACE_EXTRACTION_SYSTEM_PROMPT = """\
You are a multimodal reasoning assistant.
Your goal is to analyze the image and the question, and then produce a hidden reasoning trace that logically leads to the given answer.
The reasoning must be step-by-step, plausible, and based on the visual evidence and the question.
Do not mention, restate, or hint at the correct answer provided to you in the reasoning trace.
Output your reasoning and the correct answer using the exact format below:

```
<think> [your reasoning trace here] </think>
<answer> [the provided answer] </answer>
```\
"""

REASONING_TRACE_EXTRACTION_USER_TEMPLATE = """\
Question: {question}
Correct Answer: {answer}

Please produce a reasoning trace that could lead to this answer, but without mentioning or hinting at it explicitly.
Focus on describing visual evidence, spatial relations, object properties, and logical deductions based on what is visible.\
"""

EXAMPLE_1="""
EXAMPLE 1 with Relevant passage:

Question: What are the precocial young of this bird vulnerable to?

Retrieved relevant Passage: 
parental care is variable within birds and the kentish plover has a slightly different mechanism to other shorebirds. as discussed above, both parents incubate the eggs, however both parents do not always stick around once the eggs have hatched. it is not unusual for one parent to leave the chicks after a variable amount of time; this is referred to as brood desertion. brood desertion is the \u2018termination of care, by either one or both parents, before the offspring are capable of surviving independently\u2019 and usually occurs after one week of the brood being accompanied by both parents. brood desertion has been observed in both males and females, however females desert the brood significantly more frequently than males. \nstudies have shown that both the male and female kentish plover can provide adequate care for their brood on their own, so it is not the differences in the ability of the parents that determines which parent deserts the brood and which stays to care for the chicks. however, studies have also shown that after desertion females have a larger chance of breeding success than males, potentially due to many kentish plover populations maintaining a male-biased osr (operational sex ratio - the ratio of males actively breeding to females). therefore, is it hypothesized that the amount of reproductive success gained by desertion is what actually determines who deserts the brood,. in short, males and females can care for their brood equally, however females gain more by deserting their brood than males, resulting in a higher amount of female desertion over male. the non-deserting parent can continue to brood their chicks up to 80% of the time for over 20 days after hatching, as precocial young are vulnerable and exposed to external temperatures.\nif the parent bird feels that the eggs or chicks are under attack, then it will feign injury in order to divert attention towards itself.

Correct answer: "external temperatures"


Output:
<think> The image depicts a small shorebird with sandy-colored plumage, a black neck band, and a white underside which is characteristics of the Kentish plover. The question asks what the precocial young of this bird are vulnerable to. The retrieved passage provides detailed behavioral and ecological information about the Kentish plover. It explains that after hatching, one parent often deserts the brood, but the remaining parent can continue to provide care. The key point is that “the non-deserting parent can continue to brood their chicks up to 80% of the time for over 20 days after hatching, as precocial young are vulnerable and exposed to external temperatures.” This statement explicitly identifies that the chicks are vulnerable to external temperatures, implying that exposure to heat or cold could threaten their survival. The passage does not mention other dangers such as predators, so the relevant vulnerability is temperature-related. Thus, the text is directly relevant and provides the answer. </think>
<answer> external temperatures </answer>\
"""


EXAMPLE_2="""
EXAMPLE 2 with Irrelevant passage:

Question: What are the precocial young of this bird vulnerable to?

Retrieved irrelevant Passage: 
parental care is variable within birds and the kentish plover has a slightly different mechanism to other shorebirds. as discussed above, both parents incubate the eggs, however both parents do not always stick around once the eggs have hatched. it is not unusual for one parent to leave the chicks after a variable amount of time; this is referred to as brood desertion. brood desertion is the \u2018termination of care, by either one or both parents, before the offspring are capable of surviving independently\u2019 and usually occurs after one week of the brood being accompanied by both parents. brood desertion has been observed in both males and females, however females desert the brood significantly more frequently than males. \nstudies have shown that both the male and female kentish plover can provide adequate care for their brood on their own, so it is not the differences in the ability of the parents that determines which parent deserts the brood and which stays to care for the chicks. however, studies have also shown that after desertion females have a larger chance of breeding success than males, potentially due to many kentish plover populations maintaining a male-biased osr (operational sex ratio - the ratio of males actively breeding to females). therefore, is it hypothesized that the amount of reproductive success gained by desertion is what actually determines who deserts the brood,. in short, males and females can care for their brood equally, however females gain more by deserting their brood than males, resulting in a higher amount of female desertion over male. the non-deserting parent can continue to brood their chicks up to 80% of the time for over 20 days after hatching, as precocial young are vulnerable and exposed to external temperatures.\nif the parent bird feels that the eggs or chicks are under attack, then it will feign injury in order to divert attention towards itself.

Correct answer: "external temperatures"


Output:
<think> The image shows a bird with its young that are mobile and covered in down feathers, indicating that they are precocial—able to leave the nest shortly after hatching and feed themselves. The visual evidence highlights that the chicks are exposed on the ground or in an open environment, without full feather coverage or protection from the elements. This suggests that their survival may be influenced by environmental factors. <br>
The retrieved passage describes the spotless starling and its distribution but contains no information about the development, vulnerability, or environmental risks faced by its young. Therefore, the passage is largely irrelevant to answering the question.
From the visual evidence, the precocial chicks' exposure and incomplete feathering make them susceptible to environmental influences, particularly those related to the surrounding climate. Logical deduction from the image indicates that such young birds are at risk from changes in external conditions due to their limited physiological defenses and lack of shelter. </think>
<answer> external temperatures </answer>\
"""

REASONING_TRACE_EXTRACTION_SYSTEM_PROMPT_w_PASSAGE = """\
You are a multimodal reasoning assistant.
Your goal is to analyze the image, the question, and the retrieved passage, and then produce a hidden reasoning trace that logically leads to the given answer.
The reasoning must be step-by-step, plausible, and based on both the visual evidence and the retrieved text passage.
You MUST explicitly state, within your reasoning trace, whether the passage is relevant or not according to the information provided (i.e., if it is labeled as "irrelevant", your reasoning must clearly and logically explain why it is not relevant, and if it is labeled as "relevant", your reasoning must logically support its relevance).
Do not mention, restate, or hint at the correct answer in the reasoning trace.

Your reasoning trace should include:
- Description of relevant visual evidence (objects, spatial relations, attributes).
- Analysis of the retrieved passage (what it states, whether it supports or contradicts the image/question, and its relevance).
- Logical deduction that connects the visual and textual evidence to reach a conclusion.

Here you have two good examples of reasoning traces:
"""+EXAMPLE_1+EXAMPLE_2+"""


Output your reasoning and the correct answer using the exact format below:
```
<think> [your reasoning trace here] </think>
<answer> [the provided answer] </answer>
```\
"""

REASONING_TRACE_EXTRACTION_USER_TEMPLATE_w_PASSAGE = """\
Question: {question}
Retrieved {relevant} Passage: {passage}
Correct Answer: {answer}

Please produce a reasoning trace that could logically lead to this answer, based on both the image and the retrieved passage if relevant. 
Do not mention or hint at the answer explicitly in your reasoning.
Concentrate on providing a coherent explanation that supports the indicated relevance or irrelevance of the passage in the reasoning trace, integrating both textual and visual evidence.
Make sure to insert the correct answer between the answer tags.
"""


SYSTEM_PROMPT_REASONING_GTONLY="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant provides the user with the answer, enclosed within <answer> </answer> tags, i.e., <answer>short answer here</answer>"

