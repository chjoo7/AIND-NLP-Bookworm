# Project: Bookworm

A simple question-answering system built using IBM Watson's NLP services.

## Overview

In this project, you will use IBM Watson's NLP Services to create a simple question-answering system. You will first use the Discovery service to pre-process a document collection and extract relevant information. Then you will use the Conversation service to build a natural language interface that can respond to questions.

## Learning Objectives

By completing this project, you will learn how to:

- Create a cloud-based NLP service instance and configure it.
- Ingest a set of text documents using the service and analyze the results.
- Accept questions in natural language and parse them.
- Find relevant answers from the preprocessed text data.

## Getting Started - IBM Bluemix account creation

In order to use Watson's cloud-based services, you first need to create an account on the [IBM Bluemix platform](https://console.ng.bluemix.net/).

<div>
    <div style="display: table-cell; width: 50%;">
        <img src="images/watson-logo.png" alt="IBM Watson logo" width="200" />
    </div>
    <div style="display: table-cell; width: 50%;">
        <img src="images/bluemix-logo.png" alt="IBM Bluemix logo" width="400" />
    </div>
</div>

Then, for each service you want to use, you have to create an instance of that service. You can continue with the tasks below, and create a service instance when indicated.

## Getting Started- clone repo 

Clone this repository to your local computer.

```
git clone https://github.com/udacity/AIND-NLP-Bookworm

```

If you have the AIND Anaconda environment prepared, now is a good time to activate it.

Open the notebook `bookworm.ipynb` from a terminal using the following command:

```
jupyter notebook bookworm.ipynb
```

Then follow the instructions in the notebook.

**Note**: You may have to install some packages (mentioned in the notebook). To do so, simply open another terminal and use pip.

## 1. Create and configure Discovery service

Create an instance of the **Discovery** service. You will use this to process a set of text documents, and _discover_ relevant facts and relationships.

- Go to the [IBM Bluemix Catalog](https://console.bluemix.net/catalog/).
- Select [Discovery](https://console.bluemix.net/catalog/services/discovery) service under the [AI](https://console.bluemix.net/catalog/?category=ai) category.
- Enter a Service Name for that instance, e.g. `Discovery-Bookworm` and click **`Create`** button on the bottom right hand corner of the screen.
- You should be able to see your newly-created service in your [Bluemix Apps Dashboard](https://console.bluemix.net/dashboard/apps).
<img src="images/app-dashboard-discovery.png" alt="App Dashboard" width="800" />

- Open the `Discovery-Bookworm` service instance and find your `Url` and `API Key` in **Credentials** section.

<img src="images/discovery-apikey.png" alt="Discovery Service - Credentials tab" width="800" />

_Note: you will need the username and password when connecting to the service in the next steps shortly._

### Connect to the service instance

Let's connect to the service instance you just created using IBM Watson's [Python SDK](https://github.com/watson-developer-cloud/python-sdk). You will first need to install the SDK:
```bash
pip install watson-developer-cloud
```

Now execute each code cell below using **`Shift+Enter`**, and complete any steps indicated by a **`TODO`** comment. For more information on the Discovery service, please read the [Documentation](https://www.ibm.com/watson/developercloud/doc/discovery/index.html) and look at the [API Reference](https://www.ibm.com/watson/developercloud/discovery/api/v1/?python) as needed.

###  Using the Service Credentials

Before you can connect to Watson Service, you need to copy and paste `Username` and `Password` credentials from Bluemix Service console to this notebook.
_Note: these credentials are different from your IBM Bluemix login, and are specific to the service instance._

1. Open `service-credentials.json` file:
    * from this Jupyter Notebook top navigation, click `File` &rarr; `Open` &rarr; `service-credentials.json`
2. Copy your `API Key` and `Url` from Discovery service console.
3. Paste the credentials into `apikey` and `url` values in `Discovery` object.


<img src="images/service-discovery-json.png" alt="Discovery Service - Credentials JSON" width="600" />

### Create an environment

The Discovery service organizes everything needed for a particular application in an [_environment_](https://www.ibm.com/watson/developercloud/discovery/api/v1/curl.html?curl#environments-api). An environment must be created before collections of private data can be created.

Let's create one called `Bookworm` for this project.

> _**Note**: It is okay to run this block multiple times - it will not create duplicate environments with the same name._

There are 3 main configuration blocks that affect how input documents are processed:

1. **conversions**: How to convert documents in various formats (Word, PDF, HTML) and extract elements that indicate some structure (e.g. headings).
2. **enrichments**: What NLP output results are we interested in (keywords, entities, sentiment, etc.).
3. **normalizations**: Post-processing steps to be applied to the output. This can be left empty in most cases, unless you need the output to be normalized into a very specific format.

_**Note**: The default configuration for an environment cannot be modified. If you need to change any of the options, you will need to create a new one, and then edit it. The easiest way to do this is using the service dashboard, which is described later._

## 2. Ingest documents

### Create a collection

A _collection_ is used to organize documents of the same kind. For instance, you may want to create a collection of book reviews, or a collection of Wikipedia articles, but it may not make much sense to mix the two groups. This allows Watson to make meaningful inferences over the set of documents, find commonalities and identify important concepts.

Let's create a collection called `Story Chunks` in the Discovery service environment.

Once you have created a collection, you should be able to view it using the Discovery Service tool. Select the Discovery instance from your BlueMix dashboard.  To open, click the **`Launch tool`** button.

<img src="images/discovery-launch.png" alt="Discovery service - Manage tab" width="800" />

Here you should see the `Story Chunks` collection you just created.

<img src="images/discovery-tooling.png" alt="Discovery service - Tool showing collections" width="800" />

You can open the collection to view more details about it. If you need to modify configuration options, click the **Switch** link and create a new configuration (the default one cannot be changed).

### Add documents

Okay, now that we have everything set up, let's add a set of "documents" we want Watson to look up answers from, using the Python SDK. Note that Watson treats each "document" as a unit of text that is returned as the result of a query. But we want to retrieve a paragraph of text for each question. So, let's split each file up into individual paragraphs. We will use the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library for this purpose.

_**Note**: You could also add and manage documents in the collection using the Discovery tool, but you would have to split paragraphs up into separate files._

_**Note**: We have provided a set of files (`data/Star-Wars/*.html`) with summary plots for Star Wars movies, but you are free to use a collection of your choice. Open one of the files in a text editor to see how the paragraphs are delimited using `<p>...</p>` tags - this is how the code block below split paragraphs into separate "documents"._

## 3. Parse natural language questions

In order to understand questions posed in natural language, we'll use another AI service called [Watson Assistant](https://console.bluemix.net/catalog/services/watson-assistant-formerly-conversation) (Formerly 'Conversation'). It can be used to design conversational agents or _chatbots_ that exhibit complex behavior, but for the purpose of this project, we'll only use it to parse certain kinds of queries.

### Create a Assistant service instance

Just like you did for the Discovery service, create an instance of the Assistant service. Then launch the associated tool from the service dashboard.

- Go to the [IBM Bluemix Catalog](https://console.bluemix.net/catalog/).
- Select [Watson Assistant (formerly Conversation)](https://console.bluemix.net/catalog/services/watson-assistant-formerly-conversation) service under the [AI](https://console.bluemix.net/catalog/?category=ai) category.
- Enter a Service Name for that instance, e.g. `Assistant-Bookworm` and click **`Create`** button on the bottom right hand corner of the screen.
- You should be able to see your newly-created service in your [Bluemix Apps Dashboard](https://console.bluemix.net/dashboard/apps).
- Open the service instance and find your `Url` and `API Key` in **Credentials** section.
- Copy *API Key* and *URL* into `service-credentials.json` file in this notebook:

<img src="images/assistant-apikey.png" alt="Discovery Service - Credentials tab" width="600" />

<img src="images/assistant-cred.png" alt="Discovery Service - Credentials tab" width="600" />

### Create a Workspace from Watson Assistant console

A [_workspace_](https://www.ibm.com/watson/developercloud/assistant/api/v1/python.html?python#workspaces-api) allows you to keep all the items you need for a particular application in one place, just like an _environment_ in case of the Discovery service. 

From Watson Assistant console, please follow these steps:

1) Click `Launch Tool` to start Watson Assistant.
<img src="images/assistant-launch-tool.png" alt="Assistant service - Bookworm workspace" width="800" />

2) Click `Skills` tab on the navigation menu and click `Create new` button. 
<img src="images/assistant-create-new-skills.png" alt="Assistant service - Bookworm workspace" width="800" />

3) Create new skills called `Bookworm` with a suitable description, such as "I know a lot of stories. Ask me a question!".
<img src="images/assistant-add-dialog-skills.png" alt="Assistant service - Bookworm workspace" width="800" />

### Add intents

An [_intent_](https://www.ibm.com/watson/developercloud/assistant/api/v1/python.html?python#intents-api) is the goal or purpose of a user's input. Intent will determine the dialog flows with the users and allow Watson Assistant to provide a useful response. Please read Watson Assistant documentation on [_Planning Your Intents and Entities_](https://console.bluemix.net/docs/services/conversation/intents-entities.html#planning-your-entities).

Your task is to create a set of intents (at least 3) that capture the different kinds of questions that you want the system to answer, e.g. _who_, _what_ and _where_. Along with each intent, add a list of user examples or _utterances_ that map to that intent.

For instance, you could enter the following examples for the _where_ intent:

- Where is the Jedi temple located?
- Where was Luke born?

_Intent user examples should represent typical sentences that end users will use to interact with the application. The more examples you can provide for each intent, the better Watson Assistant will respond to the end user._

<img src="images/assistant-intents.png" alt="Assistant service - Intents listed" width="800" />

> See [**Defining intents**](https://console.bluemix.net/docs/services/conversation/intents.html#defining-intents) for a helpful video and further instructions.

### Add entities

Once you have your intents set, let's tell the service what [_entities_](https://www.ibm.com/watson/developercloud/assistant/api/v1/python.html?python#entities-api) we want it to identify. One way to do this is using the `Entities` tool on Watson Assistant console, and entering them one-by-one to the blank `My entities` page.

<img src="images/assistant-entities-blank.png" alt="Assistant service - No entities listed" width="800" />

> Go to [**Defining entities**](https://console.bluemix.net/docs/services/conversation/entities.html#defining-entities) to see how that is done.

But that can be tedious! So let's refer back to the entities that the Discovery service identified, and load them in programmatically.

As before, let's connect to the Assistant service first. Remember to enter your service credentials below.

### Design dialog flow

As a final step in creating the Assistant interface, let's design a typical dialog with a user. The most intuitive way to do this is to use the Dialog tab in the tool. Here, you can add _nodes_ that capture different stages in the dialog flow, and connect them in a meaningful way.

Go ahead and add at least 3 dialog nodes. Specify the triggers in terms of the intents and entities that you'd like to match, and an optional intermediate response like "Let me find that out for you." The actual response will be fetched by querying the Discovery service.

Here is what the dialog nodes should look like.

<img src="images/assistant_dialog_nodes.png" alt="Assistant service - Dialog nodes" width="800" />

## 4. Query document collection to fetch answers

The Discovery service includes a simple mechanism to make queries against your enriched collection of documents. But you have a lot of control over what fields are searched, how results are aggregated and values are returned.

### Process sample question

Choose a sample nautal language question to ask, and run it through the Assistant service, just like you did above when testing dialog flow.

### Query the collection

Design a query based on the information extracted above, and run it against the document collection. The sample query provided below simple looks for all the entities in the raw `text` field. Modify it to suit your needs.

Take a look at the [API Reference](https://www.ibm.com/watson/developercloud/discovery/api/v1/?python#query-collection) to learn more about the query options available, and for more guidance see this [documentation page](https://www.ibm.com/watson/developercloud/doc/discovery/using.html).

_**Note**: You may want to design different queries based on the intent / dialog node that was triggered._

### Process returned results

If you properly structure the query, Watson is able to do a pretty good job of finding the relevant information. But the result returned is a JSON object. Now your task is to convert that result into an appropriate response that best addresses the original natural language question that was asked.

E.g. if the question was "Who saved Han Solo from Jabba the Hutt?" the answer should ideally just be "The Rebels" and not the entire paragraph describing Han Solo's rescue. But that can be a backup response if you cannot be more specific.

_**Note**: You may have to go back to the previous step and modify the query, especially what you want the Discovery service to return, and this may depend on the intent / dialog node triggered. E.g. study the different parts of a "relation" structure to see how you might construct queries to match them._


## Tasks

Complete each task in the notebook by implementing or modifying code wherever there is a `TODO` comment in a code cell, and answering any inline questions by modifying markdown cells. E.g.:

> **Q**: What is the overall sentiment detected in this text? Mention the type (positive/negative) and score.
>
> **A**: Negative, -0.798

Once you have completed all tasks, save the notebook, and then export it into a PDF or HTML. Remember to submit both the notebook  (.ipynb) and the PDF/HTML, along with any other files that may be needed, e.g. data files, in case you use your own (sample files provided with the project don't need to be submitted).

**Note**: Please do not submit your `service-credentials.json` file - that is meant to be kept secret.

## Extensions

Feel free to work on the project with your own dataset. You can also turn it into a web-based application and deploy it on Bluemix.

## IBM Watson Resources

- [Watson Developer Cloud](https://www.ibm.com/watson/developercloud/) [[GitHub](https://github.com/watson-developer-cloud/)]
  - [Starter Kits](https://www.ibm.com/watson/developercloud/starter-kits.html)
  - [Discovery service](https://www.ibm.com/watson/developercloud/doc/discovery/index.html)
  - [Conversation service](https://www.ibm.com/watson/developercloud/doc/conversation/index.html)


<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
