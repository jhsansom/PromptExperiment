import openai
import json
import datetime
import os
import time
import copy
from PIL import Image, ImageDraw
import base64
from io import BytesIO

class GPTExperiment:

    def __init__(self, model='human', key=None, filename=None, **args):
        # Tracking variables
        self.total_cost = 0.0
        self.create_data_struct(model, key, **args)
        self.prompt_counter = 0

        # Constants
        self.model = model
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            filename = f'logfiles/experiment-{timestamp}.json'
        self.filename = filename
        self.save_period = 1 # save every N calls to the prompt function
        self.timeout = 10
        self.temperature = 0
        self.verbose = True
        self.remove_images = True

        # Set API key
        self.set_api_key(key)

    # Create the data structure for recording all prompts and responses
    def create_data_struct(self, model, key, **args):
        self.data = {'metadata' : args}
        self.data['metadata']['model'] = model
        self.data['metadata']['key'] = key
        self.data['metadata']['total_cost'] = 0
        self.data['data'] = []

    # Set the API key for OpenAI
    def set_api_key(self, key):
        if key is not None:
            openai.api_key = key

    def save_data(self):
        # Create the directory if it doesn't exist
        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the json file
        with open(self.filename, 'w') as fp:
            json.dump(self.data, fp, indent=4)

    def update_cost(self, usage_data):
        model_price = model_pricing[self.model]
        cost = usage_data['prompt_tokens'] / model_price['n'] * model_price['prompt'] + \
            usage_data['completion_tokens'] / model_price['n'] * model_price['completion']
        
        self.total_cost += cost
        self.data['metadata']['total_cost'] = self.total_cost
        if self.verbose:
            print(f'Cost of this prompt = ${cost:.2f}')
            print(f'Total cost = ${self.total_cost:.2f}')
            
        return cost

    # Prompt GPT and obtain a response
    # Refer to text_conversation and vision_conversation at the bottom 
    # for examples of the structure of the variable `conversation`
    def prompt(self, conversation):
        if self.model == 'human':
            (response, usage_data) = self.prompt_human(conversation)
        else:
            conversation_w_images_encoded = encode_images_enmasse(conversation)
            (response, usage_data) = self.prompt_gpt(conversation_w_images_encoded)

        if usage_data is not None:
            cost = self.update_cost(usage_data)
        else:
            cost = 0

        datapoint = {
            'conversation' : conversation,
            'response' : response,
            'cost' : cost,
            'usage_data' : usage_data
        } 
        self.data['data'].append(datapoint)

        self.prompt_counter += 1
        if (self.save_period > 0) and (self.prompt_counter % self.save_period == 0):
            self.save_data()

        return datapoint

    # Prompts the OpenAI model of your choice
    def prompt_gpt(self, conversation):
        try:
            gpt_response = openai.chat.completions.create(
                model=self.model,
                timeout=self.timeout,
                messages=conversation,
                temperature=self.temperature
            )
        except (openai.RateLimitError) as e:
            print(e)
            print('Sleeping for 60s due to OpenAI rate limiting')
            time.sleep(60)
            print('Trying same prompt again')
            return self.prompt_gpt(conversation)
        except (openai.APITimeoutError) as e:
            print(e)
            print(f'Doubling timeout from {self.timeout} to {self.timeout*2} seconds')
            self.timeout = self.timeout * 2
            print('Trying same prompt again')
            return self.prompt_gpt(conversation)
        except (openai.BadRequestError) as e:
            print('==== ERROR DURING PROMPTING ====')
            print(e)
            return (f'Error {e}', {'prompt_tokens' : 0, 'completion_tokens': 0})
        usage_data = {
            'prompt_tokens' : gpt_response.usage.prompt_tokens, 
            'completion_tokens' : gpt_response.usage.completion_tokens
        }
        response = gpt_response.choices[0].message.content

        return (response, usage_data)

    # Prompts you, the human
    def prompt_human(self, conversation):
        for item in conversation:
            print(f'======== AGENT: {item["role"]} ========')
            print(item['content'])
        print()
        response = input('User response: ')

        usage_data = {'prompt_tokens' : 0, 'completion_tokens' : 0}

        return (response, usage_data)

# Encode the image so that it can be fed to visual models
def encode_image(image_path):
    if image_path is None or (not os.path.exists(image_path)):
        raise Exception(f'Filepath {image_path} does not exist')

    with Image.open(image_path) as img:
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        data = img_buffer.read()
        encoded_data = base64.b64encode(data).decode('utf-8')

        return 'data:image/png;base64,' + encoded_data

# Encodes all images in a conversation and returns a new conversation object
def encode_images_enmasse(conversation):
    conversation_copy = copy.deepcopy(conversation)
    for turn in conversation_copy:
        for content in turn['content']:
            if content['type'] == 'image_url':
                content['image_url']['url'] = encode_image(content['image_url']['url'])
    return conversation_copy

# Keeps track of model price and saves it to each experimental output
MM = 1_000_000
model_pricing = {
    'human' : {'n' : MM, 'prompt': 0, 'completion': 0}, # cost calculated per n tokens
    'gpt-4-0125-preview' : {'n' : MM, 'prompt': 10, 'completion': 30},
    'gpt-4o-2024-05-13' : {'n' : MM, 'prompt' : 5, 'completion' : 15}
}

# List of all image models
image_models = ['gpt-4o-2024-05-13']

if __name__ == '__main__':
    # Parse input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='human') # by default, prompts you, the human
    parser.add_argument('--keypath', type=str, default='./API_KEY.txt') # path to text file containing your API key
    parser.add_argument('--exampleparam', type=int, default=5)
    args = vars(parser.parse_args())
    
    # Get the key
    with open(args['keypath'], 'r') as fp: args['key'] = fp.read()

    # Initialize an experiment
    exp = GPTExperiment(**args)

    # Set up an example conversation
    text_conversation = [
        {'role' : 'user', 'content' : 'user prompt input etc.'},
        {'role' : 'assistant', 'content' : 'this was GPTs response'},
        {'role' : 'user', 'content' : 'now do this extra thing GPT...'}
    ]
    vision_conversation = [
        {'role' : 'user', 'content' : [{'type' : 'text', 'text' : 'Im going to give you an image and I want you to tell me what it is an image of'}]},
        {'role' : 'assistant', 'content' : [{'type' : 'text', 'text' : 'Ok'}]},
        {'role' : 'user', 'content' : [{'type' : 'image_url', 'image_url' : {'url' : './sample-image.png'}}]}
    ]

    # Prompt the model with either text_conversation or vision_conversation
    if args['model'] in image_models:
        conversation = vision_conversation
    else:
        conversation = text_conversation

    # Prompt the model
    exp.prompt(conversation)
    