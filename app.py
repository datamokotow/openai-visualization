import pandas as pd
import openai
import streamlit as st

def run_request(question_to_ask, model_type, key):
    if model_type == "gpt-4":
        task = "Generate Python Code Script. The script should only include code, no comments."
    elif model_type == "gpt-3.5-turbo":
        task = "Generate Python Code Script."
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo":
        # Run ChatGPT API
        openai.api_key = key
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {"role":"system","content":task},
                {"role":"user","content":question_to_ask}])
        res = response["choices"][0]["message"]["content"]
    else:
        # We use the API to submit the query
        openai.api_key = key
        response = openai.Completion.create(
            # model_type = text-davinci-003, code-davinci-002
            engine=model_type,
            prompt=question_to_ask,
            temperature=0,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["plt.show()"]
            )
        res = response["choices"][0]["text"] 
    # rejig the response
    res = format_response(res)
    return res

def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question):
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code

#from classes import get_primer,format_question,run_request
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide",page_title="Data Visualization")
st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>DataMokotow | Creating Visualizations using LLMs \
             </h2>", unsafe_allow_html=True)

available_models = {"ChatGPT-4": "gpt-4","ChatGPT-3.5": "gpt-3.5-turbo"}

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Cars"] =pd.read_csv("C:/Users/Rutvi/OneDrive/2. Documents/GenAI/Chat2Vis/cars.csv")
    datasets["Energy Production"] =pd.read_csv("C:/Users/Rutvi/OneDrive/2. Documents/GenAI/Chat2Vis/energy_production.csv")
    st.session_state["datasets"] = datasets
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

with st.sidebar:
    # Input the OpenAI key
    my_key = st.text_input(label = ":key: OpenAI Key:",type="password")
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
    index_no=0
    if uploaded_file is not None:
        # Read in the data, add it to the list of available datasets
        file_name = uploaded_file.name[:-4].capitalize()
        datasets[file_name] = pd.read_csv(uploaded_file)
        # Default for the radio buttons
        index_no = len(datasets)-1

    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)#,horizontal=True,)

    # Check boxes for model choice
    st.write(":brain: Choose your model(s):")
    # Keep a dictionary of whether models are selected or not
    use_model = {}
    for model_desc,model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label,value=True,key=key)

# Text area for query
question = st.text_area("Enter your visualization query")
go_btn = st.button("Submit")

# Make a list of the models which have been selected
model_list = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(model_list)

# Execute chatbot query


# ... (existing code for imports and functions)

# Execute chatbot query
if go_btn and model_count > 0:
    # Place for plots depending on how many models
    plots = st.columns(model_count)
    # Get the primer for this dataset
    primer1,primer2 = get_primer(datasets[chosen_dataset],'datasets["'+ chosen_dataset + '"]')
    # Format the question
    question_to_ask = format_question(primer1,primer2 , question)    
    # Create model, run the request and print the results
    for plot_num, model_type in enumerate(model_list):
        with plots[plot_num]:
            st.subheader(model_type)
            try:
                # Run the question
                answer=""
                answer = run_request(question_to_ask, available_models[model_type], key=my_key)
                # the answer is the completed Python script so add to the beginning of the script to it.
                answer = primer2 + answer
                plot_area = st.empty()
                plot_area.pyplot(exec(answer))     
                # Display the generated code
                st.text("Generated Visualization Code:")
                st.code(answer, language='python')      
            except Exception as e:
                if type(e) == openai.error.APIError:
                    st.error("OpenAI API Error. Please try again a short time later.")
                elif type(e) == openai.error.Timeout:
                    st.error("OpenAI API Error. Your request timed out. Please try again a short time later.")
                elif type(e) == openai.error.RateLimitError:
                    st.error("OpenAI API Error. You have exceeded your assigned rate limit.")
                elif type(e) == openai.error.APIConnectionError:
                    st.error("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings.")
                elif type(e) == openai.error.InvalidRequestError:
                    st.error("OpenAI API Error. Your request was malformed or missing required parameters.")
                elif type(e) == openai.error.AuthenticationError:
                    st.error("Please enter a valid OpenAI API Key.")
                elif type(e) == openai.error.ServiceUnavailableError:
                    st.error("OpenAI Service is currently unavailable. Please try again a short time later.")                   
                else:
                    st.error("Unfortunately the code generated from the model contained errors and was unable to execute. ")
                
# ... (remaining code for displaying datasets and social media links)


# Display the datasets in a list of tabs
# Create the tabs
tab_list = st.tabs(datasets.keys())

# Load up each tab with a dataset
for dataset_num, tab in enumerate(tab_list):
    with tab:
        # Can't get the name of the tab! Can't index key list. So convert to list and index
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name],hide_index=True)

# social media links added to the sidebar
st.sidebar.markdown('**Connect**')
st.sidebar.markdown('[GitHub](https://github.com/datamokotow)')
st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/rutvikacharya/)')
st.sidebar.markdown('[Twitter](https://twitter.com/datamokotow)')

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
