import openai
import requests 
import json
from openai import AzureOpenAI
from collections import deque
from datetime import datetime

import sys
from io import StringIO



# Azure OpenAI Configuration
azure_openai_client = AzureOpenAI(
    api_key="6133f9e8a55842fcba2b074a8a8ca463",
    api_version="2024-08-01-preview",
    azure_endpoint="https://serrizero.openai.azure.com/"
)

# Configuration for Azure OpenAI API
API_KEY = ""
AZURE_ENDPOINT = "https://serrizero.openai.azure.com/"

# Headers for Azure OpenAI API
openai.api_key = API_KEY
openai.api_base = AZURE_ENDPOINT
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"

from collections import deque
from datetime import datetime

class OutputCapture:
    def __init__(self, history_size=5):
        self.history = []  # Using a simple list to store prompt-output pairs.
        self.history_size = history_size

    def capture_interaction(self, prompt, response):
        """Stores each prompt and response as a pair."""
        # Add the interaction to history
        self.history.append({'prompt': prompt, 'response': response})
        # Truncate history to maintain a max length of history_size
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
    
    def show_history(self):
        """Displays the last few interactions up to the specified history size."""
        print("\n=== Last Interactions ===")
        for entry in self.history[-self.history_size:]:
            print(f"\nPrompt: {entry['prompt']}\nResponse:\n{entry['response']}")
        print("==========================\n")


class ChatMemory:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.parameter_history = {}
        self.prompt_history = deque(maxlen=history_size)
        self.common_parameters = {
            'project_id': None,
            'api_key': None,
            'campaign_id': None
        }
    
    def add_prompt(self, prompt, parameters):
        """Stores prompts with parameters and common parameters."""
        timestamp = datetime.now()
        self.prompt_history.append({
            'timestamp': timestamp,
            'prompt': prompt,
            'parameters': parameters
        })
        
        # Update common parameters if provided
        for key in self.common_parameters:
            if key in parameters:
                self.common_parameters[key] = parameters[key]
        
        # Track parameter values for each key
        for key, value in parameters.items():
            if key not in self.parameter_history:
                self.parameter_history[key] = deque(maxlen=self.history_size)
            if value not in self.parameter_history[key]:
                self.parameter_history[key].append(value)
    
    def get_last_value(self, parameter):
        """Returns the latest value of a parameter, either from common_parameters or history."""
        return self.common_parameters.get(parameter) or (
            self.parameter_history.get(parameter) and self.parameter_history[parameter][-1]
        )
    
    def get_parameter_history(self, parameter):
        """Returns the history of a specific parameter."""
        return list(self.parameter_history.get(parameter, []))

    def has_recent_value(self, parameter):
        """Check if parameter has a recent value."""
        return bool(self.get_last_value(parameter))


# Initialize memory
chat_memory = ChatMemory()

def get_openai_context(output_capture):
    # Combine the last history_size interactions to build context
    context = "\n".join([
        f"User: {entry['prompt']}\nAssistant: {entry['response']}"
        for entry in output_capture.history[-output_capture.history_size:]
    ])
    return context


def infer_intent(user_input):
    """Infers the user's intent using AI."""
    prompt = f"Classify the prompt '{user_input}' into one of the 6 categories: 1) GetProjectDetails, 2) SendCampaign, 3) SubmitTemplate, 4) ListTemplates, 5) GetCampaignAnalytics, or 6) NaturalLanguageQuery (when the user is asking a general question about previous prompts or results). RETURN ONLY THE NAME OF THE INTENT, NOT THE INDEX  "
    response = azure_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that classifies the user prompt into one of the 6 categories: 1) GetProjectDetails, 2) SendCampaign, 3) SubmitTemplate, 4) ListTemplates, 5) GetCampaignAnalytics, or 6) NaturalLanguageQuery (when the user is asking a general question about previous prompts or results)"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    intent = response.choices[0].message.content.strip()
    return intent

def get_required_parameters(intent):
    """Returns a list of required parameters based on intent."""
    if intent == "SubmitTemplate":
        return ["project_id", "api_key", "label", "category", "type", "language", "name", "text", "sample_text", "message_action_type"]
    elif intent == "SendCampaign":
        return ["project_id", "api_key", "default_country_code", "name", "phone_number", "template_params", "campaign_name"]
    elif intent == "GetProjectDetails":
        return ["project_id", "api_key"]
    elif intent == "ListTemplates":
        return ["project_id", "api_key"]
    elif intent == "GetCampaignAnalytics":
        return ["project_id", "api_key", "campaign_id"]
    else:
        raise ValueError(f"Unknown intent: {intent}")

def collect_missing_parameters(parameters, required_params):
    """Prompts the user for missing parameters with memory support."""
    for param in required_params:
        if param not in parameters:
            last_value = chat_memory.get_last_value(param)
            if last_value:
                use_previous = input(f"Use previous value for '{param}' ({last_value})? (yes/no): ")
                if use_previous.lower() == 'yes':
                    parameters[param] = last_value
                    continue
            
            # Show parameter history if available
            param_history = chat_memory.get_parameter_history(param)
            if param_history:
                print(f"\nPrevious values for '{param}':")
                for idx, value in enumerate(param_history, 1):
                    print(f"{idx}. {value}")
                print("Enter new value or select number from history:")
            
            value = input(f"Please provide the value for '{param}': ")
            
            # Check if user selected from history
            try:
                idx = int(value) - 1
                if 0 <= idx < len(param_history):
                    value = param_history[idx]
            except ValueError:
                pass
            
            parameters[param] = value
    
    # Add parameters to memory
    chat_memory.add_prompt("Parameter Collection", parameters)
    return parameters

def get_template_category():
    """Get template category from user input."""
    categories = ['MARKETING', 'AUTHENTICATION', 'UTILITY']
    while True:
        print("\nAvailable categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        choice = input("Select template category (1-3): ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(categories):
                return categories[index]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def get_template_type():
    """Get template type from user input."""
    types = ['TEXT', 'MEDIA', 'INTERACTIVE']
    while True:
        print("\nAvailable types:")
        for i, type_ in enumerate(types, 1):
            print(f"{i}. {type_}")
        choice = input("Select template type (1-3): ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(types):
                return types[index]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def get_template_language():
    """Get template language from user input."""
    languages = ['English', 'Gujarati', 'Polish', 'Marathi', 'Hindi']
    while True:
        print("\nAvailable languages:")
        for i, lang in enumerate(languages, 1):
            print(f"{i}. {lang}")
        choice = input("Select template language (1-5): ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(languages):
                return languages[index]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def get_basic_template_details():
    """Collect basic template details from user."""
    print("\nLet's gather some basic details about your template:")
    
    details = {
        'label': input("Enter template label (internal reference): "),
        'category': get_template_category(),
        'type': get_template_type(),
        'language': get_template_language(),
        'name': input("Enter template name (visible to customers): "),
        'purpose': input("Describe the purpose of this template: ")
    }
    return details

def recommend_template(details):
    """Use AI to recommend a template based on provided details."""
    prompt = f"""Generate a WhatsApp template message based on these details:
    Category: {details['category']}
    Type: {details['type']}
    Language: {details['language']}
    Purpose: {details['purpose']}

    Requirements:
    1. Must be professional and compliant with WhatsApp policies.
    2. Should be brief and clear.
    3. Use placeholders like {{1}} for variable content in 'text' and [Name] in 'sample_text' format.
    4. If type is INTERACTIVE, suggest quick reply options or call-to-action.

    Return the response in this JSON format:
    {{
      "label": "{details['label']}",
      "category": "{details['category']}",
      "type": "{details['type']}",
      "language": "{details['language']}",
      "name": "{details['name']}",
      "text": "Hello {{1}},\\n\\n------USE YOUR GENERATED CONTENT HERE--KEEP VARIABLES IN {{-}}---\\n{{2}}\\n\\nBest regards,\\n{{3}}",
      "sample_text": "Hello [Name],\\n\\---------USE YOUR GENERATED CONTENT HERE----KEEP VARIABLES IN [-]---\\n[https://google.com]\\n\\nBest regards,\\n[Serri Events]",
      "message_action_type": "NONE"
    }}
    """

    response = azure_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a WhatsApp template specialist who creates professional and compliant template messages. Use placeholders like {{--}} for variable content in 'text' and [--] in 'sample_text' format. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,

    )

    try:
        recommendation = json.loads(response.choices[0].message.content.strip())
        return recommendation
    except Exception as e:
        print(f"Error parsing recommendation: {e}")
        return None

def confirm_template(recommendation):
    """Show recommendation to user and allow modifications."""
    print("\nRecommended Template:")
    print(f"Text: {recommendation['text']}")
    print(f"Sample: {recommendation['sample_text']}")
    print(f"Action Type: {recommendation['message_action_type']}")
    
    if recommendation.get('quick_replies'):
        print("Quick Replies:", recommendation['quick_replies'])
    if recommendation.get('call_to_action'):
        print("Call to Action:", recommendation['call_to_action'])

    while True:
        choice = input("\nDo you want to:\n1. Use this template\n2. Modify the template\nChoice (1-2): ")
        
        if choice == "1":
            return recommendation
        elif choice == "2":
            recommendation['text'] = input("Enter modified template text: ")
            recommendation['sample_text'] = input("Enter modified sample text: ")
            if recommendation['message_action_type'] == 'QUICK_REPLY':
                quick_replies = []
                while True:
                    reply = input("Enter quick reply option (or press enter to finish): ")
                    if not reply:
                        break
                    quick_replies.append(reply)
                recommendation['quick_replies'] = quick_replies
            elif recommendation['message_action_type'] == 'CALL_TO_ACTION':
                cta_type = input("Enter call to action type (PHONE_NUMBER or URL): ")
                cta_value = input("Enter call to action value: ")
                recommendation['call_to_action'] = {
                    "type": cta_type,
                    "value": cta_value
                }
            return recommendation

def generate_json_workflow(intent, parameters):
    """Generates the JSON workflow based on intent and parameters."""
    headers = {
        "X-AiSensy-Project-API-Pwd": parameters["api_key"]
    }
    
    if intent == "SubmitTemplate":
        workflow = {
            "method": "POST",
            "url": f"https://apis.aisensy.com/project-apis/v1/project/{parameters['project_id']}/wa_template",
            "headers": headers,
            "body": {
                "label": parameters["label"],
                "category": parameters["category"],
                "type": parameters["type"],
                "language": parameters["language"],
                "name": parameters["name"],
                "text": parameters["text"],
                "sample_text": parameters["sample_text"],
                "message_action_type": parameters["message_action_type"]
            }
        }
        
        if parameters.get("quick_replies"):
            workflow["body"]["quick_replies"] = parameters["quick_replies"]
        if parameters.get("call_to_action"):
            workflow["body"]["call_to_action"] = parameters["call_to_action"]
            
        return workflow
    
    elif intent == "SendCampaign":
        return {
            "method": "POST",
            "url": f"https://apis.aisensy.com/project-apis/v1/project/{parameters['project_id']}/campaign/api/send",
            "headers": headers,
            "body": {
                "default_country_code": parameters["default_country_code"],
                "name": parameters["name"],
                "phone_number": parameters["phone_number"],
                "template_params": parameters["template_params"],
                "campaign_name": parameters["campaign_name"],
                "campaign_description": parameters["campaign_description"]
            }
        }
    elif intent == "GetProjectDetails":
        return {
            "method": "GET",
            "url": f"https://apis.aisensy.com/project-apis/v1/project/{parameters['project_id']}",
            "headers": headers
        }
    elif intent == "ListTemplates":
        return {
            "method": "GET",
            "url": f"https://apis.aisensy.com/project-apis/v1/project/{parameters['project_id']}/wa_template/",
            "headers": headers
        }
    elif intent == "GetCampaignAnalytics":
        return {
            "method": "POST",
            "url": f"https://apis.aisensy.com/project-apis/v1/project/{parameters['project_id']}/campaign/analytics/{parameters['campaign_id']}",
            "headers": headers,
            "body": {}
        }
    else:
        raise ValueError("Unknown intent")

def execute_workflow(workflow):
    """Executes the API call based on the generated workflow."""
    response = requests.request(
        method=workflow["method"],
        url=workflow["url"],
        headers=workflow["headers"],
        json=workflow.get("body", {})
    )
    return response.json()
def submit_template_workflow(parameters):
    """Enhanced template submission workflow."""
    # Get basic details
    template_details = get_basic_template_details()
    
    # Get AI recommendation
    print("\nGenerating template recommendation...")
    recommendation = recommend_template(template_details)
    
    if not recommendation:
        print("Failed to generate recommendation. Please try again.")
        return None
    
    # Get user confirmation or modifications
    final_template = confirm_template(recommendation)
    
    # Prepare final parameters
    parameters.update(template_details)
    parameters.update(final_template)
    
    # Generate workflow
    workflow = generate_json_workflow("SubmitTemplate", parameters)
    
    return workflow

import pandas as pd  # Make sure to install pandas if you haven't already

import pandas as pd

def send_campaign_workflow(parameters):
    """Automated multi-step campaign creation process for sending a campaign to contacts from CSV."""
    
    # Step 1: Basic Campaign Details (Auto-populate without user input)
    parameters["name"] = "Buddy"  # Set a default name or modify as desired
    parameters["campaign_name"] = input("Campaign Name: ")
    parameters["campaign_description"] = input("Campaign description: ")
    parameters["project_id"] = default_parameters["project_id"]
    parameters["api_key"] = default_parameters["api_key"]

    # Step 2: Recipient Selection - Always use CSV for contacts
    csv_file_path = input("Enter the file path of the CSV with contacts: ")
    try:
        # Load and validate the CSV file
        df = pd.read_csv(csv_file_path)
        if "phone_number" not in df.columns:
            print("CSV file must contain a 'phone_number' column.")
            return

        contact_list = df["phone_number"].tolist()
        print(f"Successfully loaded {len(contact_list)} contacts from the CSV.")

        # Other Campaign Settings - Automatically set or prompt if necessary
        parameters["default_country_code"] = input("Enter default country code (e.g., +1): ")
        parameters["template_params"] = input("Enter any template parameters (JSON format if applicable): ")

        # Confirm the campaign details
        print("\nCampaign details:")
        print(f"Name: {parameters['campaign_name']}")
        print(f"Description: {parameters['campaign_description']}")
        print(f"Recipients: {len(contact_list)} contacts from uploaded CSV")
        print(f"Country Code: {parameters['default_country_code']}")
        print(f"Template Parameters: {parameters['template_params']}")

        confirm = input("\nDo you want to proceed with this campaign setup? (yes/no): ")
        if confirm.lower() != "yes":
            print("Campaign setup cancelled.")
            return

        # Execute the campaign for each contact individually
        for phone_number in contact_list:
            parameters["phone_number"] = phone_number
            workflow = generate_json_workflow("SendCampaign", parameters)
            print(workflow)
            execute_workflow(workflow)
            print(f"Campaign sent to {phone_number}")

    except FileNotFoundError:
        print("The specified file was not found. Please try again.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")


    # Save the completed campaign to memory for reference in future prompts
    chat_memory.add_prompt("Send Campaign Setup", parameters)


def chatbot():
    """Main chatbot function with memory and output capture support."""
    print("Welcome to the AiSensy-powered chatbot with memory and interaction history!")
    print("\nAvailable commands:")
    print("1. Submit a template")
    print("2. Send a campaign")
    print("3. Get project details")
    print("4. List templates")
    print("5. Get campaign analytics")
    print("Type 'history' to see the last 5 interactions")
    print("Type 'exit' to quit")
    
    output_capture = OutputCapture(history_size=5)  # Initialize output capture

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        if user_input.lower() == 'history':
            output_capture.show_history()
            continue
            
        # Infer user intent
        intent = infer_intent(user_input)
        
        if intent == "NaturalLanguageQuery":
            response = handle_natural_language_query(user_input, output_capture)
            output_capture.capture_interaction(user_input, response)
            print(response)
            continue
        elif intent == "Unknown":
            response = "I'm sorry, I couldn't understand your request. Please try again."
            output_capture.capture_interaction(user_input, response)
            print(response)
            continue

        # Initialize parameters dictionary for campaign details
        parameters = {}

        # Handle specific intent workflows
        if intent == "SendCampaign":
            send_campaign_workflow(parameters)  # Call the multi-step workflow for SendCampaign
            

        # Other intents and workflows remain the same...


        # Rest of the chatbot function remains unchanged...


        # Handle other intents and workflows as before
        # Rest of the chatbot code remains unchanged...


        # Get required parameters for the intent
        required_params = get_required_parameters(intent)
        
        # Initialize parameters with common values from memory
        parameters = {
            param: chat_memory.get_last_value(param)
            for param in required_params
            if chat_memory.has_recent_value(param)
        }
        
        # Collect remaining parameters
        if intent == "SubmitTemplate":
            workflow = submit_template_workflow(parameters)
        else:
            parameters = collect_missing_parameters(parameters, required_params)
            workflow = generate_json_workflow(intent, parameters)

        if workflow:
            def execute_and_store_workflow():
                """Executes and stores workflow output."""
                print("\nGenerated Workflow:")
                print(json.dumps(workflow, indent=2))
                
                confirm = input("\nDo you want to execute this workflow? (yes/no): ")
                if confirm.lower() == 'yes':
                    try:
                        print("\nExecuting workflow...")
                        response = execute_workflow(workflow)
                        response_json = json.dumps(response, indent=2)
                        
                        # Capture interaction for the workflow execution
                        output_capture.capture_interaction(user_input, response_json)
                        print("\nResponse:", response_json)
                        
                        # Add successful workflow to memory
                        chat_memory.add_prompt(user_input, parameters)
                        print("\nParameters have been saved to memory.")
                    except Exception as e:
                        error_message = f"\nError executing workflow: {e}"
                        output_capture.capture_interaction(user_input, error_message)
                        print(error_message)
                else:
                    print("\nWorkflow execution cancelled.")

            # Directly call the execute_and_store_workflow without capture_output
            execute_and_store_workflow()


def handle_natural_language_query(user_input, output_capture):
    """Handles natural language queries by searching recent chat history."""
    # Prepare chat history context for the model
    recent_history = get_openai_context(output_capture)  # Use recent context

    # Formulate the prompt for the OpenAI model
    prompt = f"""Using the recent chat history, respond to the following query from the user:
    
    Chat History:
    {recent_history}
    
    User Query: "{user_input}"
    
    Provide a detailed answer based on the recent chat history."""
    
    # Call Azure OpenAI with the formulated prompt
    response = azure_openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that uses recent chat history to answer user queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    # Return the response content as a string
    return response.choices[0].message.content.strip()




if __name__ == "__main__":
    # Initialize with some default values if available
    default_parameters = {
        'project_id': input("Please provide your Project_id: "),
        'api_key': input("Please provide your api_key: ")
    }
    chat_memory.add_prompt("Default initialization", default_parameters)
    
    try:
        chatbot()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")



