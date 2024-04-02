import openai
import json
import datetime
import os

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
        self.timeout = 5
        self.temperature = 0
        self.verbose = True

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
        if self.verbose:
            print(f'Cost of this prompt = ${cost:.2f}')
            print(f'Total cost = ${self.total_cost:.2f}')
            
        return cost

    # Prompt GPT and obtain a response
    # conversation is a list of dictionaries of the following form:
    # [{'role' : 'user', 'content' : 'user prompt input etc.'},
    #  {'role' : 'assistant', 'content' : 'this was GPTs response',
    #  {'role' : 'user', 'content' : 'now do this extra thing GPT...'}]
    def prompt(self, conversation):
        if self.model == 'human':
            (response, usage_data) = self.prompt_human(conversation)
        else:
            (response, usage_data) = self.prompt_gpt(conversation)

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

    # Prompts the OpenAI model of your choice
    def prompt_gpt(self, conversation):
        try:
            gpt_response = openai.chat.completions.create(
                model=self.model,
                timeout=self.timeout,
                messages=conversation,
                temperature=self.temperature
            )
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
            print(e)
            print('Sleeping for 60s due to OpenAI rate limiting')
            time.sleep(60)
            return (None, None)
        except openai.error.InvalidRequestError as e:
            print('Prompt is too long. Skipping this prompt')
            return (None, None)
        usage_data = gpt_response['usage']
        response = gpt_response['choices'][0]['message']['content']

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

MM = 1_000_000
model_pricing = {
    'human' : {'n' : MM, 'prompt': 0, 'completion': 0}, # cost calculated per n tokens
    'gpt-4-0125-preview' : {'n' : MM, 'prompt': 10, 'completion': 30}
}

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='human') # by default, prompts you, the human
    parser.add_argument('--num', type=int, default=1) # number of task descriptions
    parser.add_argument('--m2wpath', type=str, default=MIND2WEB_PATH) # path to Mind2Web trainining folder
    parser.add_argument('--keypath', type=str, default='./API_KEY.txt') # path to text file containing your API key
    parser.add_argument('--examples', type=int, default=5)
    args = vars(parser.parse_args())
    
    # Get the key
    with open(args['keypath'], 'r') as fp: args['key'] = fp.read()

    # Initialize an experiment
    exp = GPTExperiment(**args)

    # Set up this code to run your experiment
    # for i in range(...):
    #   exp.prompt(conversation)
    # ...