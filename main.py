import ollama
import requests
from extension import ExtensionInterface

class WebSearch(ExtensionInterface):

    def inference(self, newprompt, genparams, max_context, *args):
        responses = newprompt.split('### Response:')
        if len(responses) < 2:
            return newprompt
        response = responses[-1]
        if len(response) == 0:
            return newprompt
        print(f'Response: {response}')
        is_search = response.find('[search:')
        if is_search == -1:
            return newprompt

        query = response.split('[search:')[1].split(']')[0].strip()
        query = query.replace(' ', '+')
        print(f'Query: {query}')
        json_object = requests.get(f'https://api.duckduckgo.com/?q={query}&format=json').json()
        result = json_object['Abstract'] if len(json_object['Abstract']) > 0 else json_object['RelatedTopics'][0]['Text']
        result_length = len(result)
        max_new_prompt = max_context - result_length

        newprompt = (newprompt[-(max_new_prompt):]) if len(newprompt) > max_new_prompt else newprompt
        return f'Search result: [{result}] {newprompt}'


def generate_search_term(prompt):
    response = ollama.chat(model='voom', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']


def summarize_search_results(results):
    response = ollama.chat(model='voom', messages=[
        {
            'role': 'user',
            'content': results,
        },
    ])
    return response['message']['content']


def main():
    while True:
        user_prompt = input("Enter a prompt: ")
        print(f"User prompt: {user_prompt}")

        # Generate search term using ollama
        search_term_prompt = f"Generate a search term for the prompt: {user_prompt}"
        print("Generating search term using ollama...")
        search_term = generate_search_term(search_term_prompt)
        print(f"Generated search term: {search_term}")

        # Perform web search
        web_search_instance = WebSearch()
        web_search_prompt = f"### Response: [search:{search_term}]"
        search_results = web_search_instance.inference(web_search_prompt, None, 5000)
        print(f"Search results: {search_results}")

        # Summarize the search results using ollama
        summarize_prompt = f"Please summarize the following information: i {search_results}"
        print("Summarizing search results using ollama...")
        summary = summarize_search_results(summarize_prompt)
        print(f"Summary: {summary}")

        # Ask for another prompt
        if input("Would you like to enter another prompt? (yes/no): ").strip().lower() != 'yes':
            break


if __name__ == "__main__":
    main()
