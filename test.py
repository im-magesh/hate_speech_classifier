from infer import inference
from flask import Flask, request, jsonify, render_template



if __name__ == "__main__":
    sentence : str = "All these gays are son of the bitches. Mother fuckers never got any pussy in their life so they roaming catching dicks. Man I wish the days were old, were we could have electrocute these mother fuckers. We need to treat these fagots like the way Nazis treated Jews."
    result : dict = inference(sentence=sentence)
    print(result)
