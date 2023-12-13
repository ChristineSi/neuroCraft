import openai
import os

# Model implementation for text simplification
class TextSimplificationModel:
    def simplify_text(self, text):
        # Implement your text simplification logic here
        # Return the simplified text

        openai.api_key  = os.getenv("OPENAI_API_KEY")

        def get_completion(prompt, model="gpt-3.5-turbo"):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message["content"]

        text = f"""{text}"""

        prompt = f"""
        These are the criteria for the text difficulty level:

        Here is an example of an easy text:
        "The boys left the capitol and made their way down the long hill to the main business part of the town. \
        As they struck onto the main business street, Garry noticed the familiar blue bell sign of the telephone company.\
        "Say, boys, I have an idea. Let's stop in here and put in long distance calls and say hello to our folks. \
        How does the idea strike you?" said Garry, almost in one breath.\
        "Ripping," shouted Phil, while Dick didn't wait to make any remark, but dived in through the door, \
        and in a trice was putting in his call. Phil followed suit, while Garry waited, as he would talk when Dick had finished.\
        This pleasant duty done, they went to a restaurant for dinner. Here they attracted no little attention, \
        for their khaki clothes looked almost like uniforms. Added to this was the fact that they wore forest shoepacks, \
        those high laced moccasins with an extra leather sole, and felt campaign hats."

        Here is an example of an intermediate text :
        "On the twenty-second of February, 1916, an automobile sped northward along the French battle line that \
        for almost two years had held back the armies of the German emperor, strive as they would to win their way \
        farther into the heart of France. For months the opposing forces had battled to a draw from the North Sea to \
        the boundary of Switzerland, until now, as the day waned—it was almost six o'clock—the hands of time drew closer \
        and closer to the hour that was to mark the opening of the most bitter and destructive battle of the war, up to this time.
        It was the eve of the battle of Verdun. The occupants of the automobile as it sped northward numbered three. \
        In the front seat, alone at the driver's wheel, a young man bent low. He was garbed in the uniform of a British lieutenant of cavalry. \
        Close inspection would have revealed the fact that the young man was a youth of some eighteen years, fair and good to look upon."

        Here is an example of a hard text :
        "This Pedrarias was seventy-two years old. He was of good birth and rich, \
        and was the father of a large and interesting family, which he prudently left behind him in Spain. \
        His wife, however, insisted on going with him to the New World. \
        Whether or not this was a proof of wifely devotion—and if it was, it is the only thing in history \
        to his credit—or of an unwillingness to trust Pedrarias out of her sight, which is more likely, is not known. \
        At any rate, she went along. Pedrarias, up to the time of his departure from Spain, \
        had enjoyed two nick-names, El Galan and El Justador. He had been a bold and dashing cavalier in his youth, \
        a famous tilter in tournaments in his middle age, and a hard-fighting soldier all his life. His patron was Bishop Fonseca. \
        Whatever qualities he might possess for the important work about to be devolved upon him would be developed later."

        Please simplify {text} basing yourself on the following 3 examples, which can be considered as a benchmark for dyslexic readers.
        Please use these guidelines: use short and simple sentences (60-70 characters), prefer active voice,
        avoid unnecessary jargon, keep almost the same words number, include only the simplified text in your response.

        These are illustrations of the simplified level of the text:

        simple text example 1 :
        "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.
        The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.
        At each end of the room, on the wall, hung a beautiful bear-skin rug.
        These rugs were for prizes, one for the girls and one for the boys. And this was the game.
        The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.
        This would have been an easy matter, but each traveller was obliged to wear snowshoes."

        simple text example 2 :
        "All through dinner time, Mrs. Fayre was somewhat silent, her eyes resting on Dolly with a wistful, uncertain expression. She wanted to give the child the pleasure she craved, but she had hard work to bring herself to the point of overcoming her own objections.
        At last, however, when the meal was nearly over, she smiled at her little daughter, and said, "All right, Dolly, you may go."
        "Oh, mother!" Dolly cried, overwhelmed with sudden delight. "Really?
        Oh, I am so glad! Are you sure you're willing?"
        "I've persuaded myself to be willing, against my will," returned Mrs. Fayre, whimsically. "I confess I just hate to have you go, but I can't bear to deprive you of the pleasure trip. And, as you say, it would also keep Dotty at home, and so, altogether, I think I shall have to give in."
        "Oh, you angel mother! You blessed lady! How good you are!" And Dolly flew around the table and gave her mother a hug that nearly suffocated her."

        simple text example 3 :
        Once upon a time there were Three Bears who lived together in a house of their own in a wood. \
        One of them was a Little, Small, Wee Bear; and one was a Middle-sized Bear, and the other was a Great, Huge Bear. \
        They had each a pot for their porridge; a little pot for the Little, Small, Wee Bear; \
        and a middle-sized pot for the Middle Bear; and a great pot for the Great, Huge Bear. \
        And they had each a chair to sit in; a little chair for the Little, Small, Wee Bear; \
        and a middle-sized chair for the Middle Bear; and a great chair for the Great, Huge Bear. \
        And they had each a bed to sleep in; a little bed for the Little, Small, Wee Bear; \
        and a middle-sized bed for the Middle Bear; and a great bed for the Great, Huge Bear.

        \"\"\"{text}\"\"\"
        """
        response = get_completion(prompt)
        return response

if __name__ == "__main__":
    # Testing the model with a sample text
    text = f"""
    It was believed by the principal men of Virginia that Talbot's sympathies were with the revolted colonies; \
    but the influence of his mother, to whom he had been accustomed to defer, \
    had hitherto proved sufficient to prevent him from openly declaring himself. \
    His visit to England, and the delightful reception he had met with there, \
    had weakened somewhat the ties which bound him to his native country, \
    and he found himself in a state of indecision as humiliating as it was painful. \
    Lord Dunmore and Colonel Wilton had each made great efforts to enlist his support, \
    on account of his wealth and position and high personal qualities. \
    It was hinted by one that the ancient barony of the Talbots would be revived by the king; \
    and the gratitude of a free and grateful country, \
    with the consciousness of having materially aided in acquiring that independence which should be the birthright of every Englishman, \
    was eloquently portrayed by the other. When to the last plea was added the personal preference of Katharine Wilton, \
    the balance was overcome, and the hopes of the mother were doomed to disappointment.
    """
    print(TextSimplificationModel().simplify_text(text = text))
