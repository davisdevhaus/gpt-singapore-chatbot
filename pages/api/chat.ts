import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAI } from 'langchain/llms/openai';
import { makePineconeChain } from '@/utils/chains/makePineconeChain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} from 'langchain/prompts';
import { SerpAPI } from 'langchain/tools';
import { AgentExecutor, ChatAgent } from 'langchain/agents';
import { ConversationChain } from 'langchain/chains';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history } = req.body;

  //only accept post requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    /* create vectorstore*/
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings({}),
      {
        pineconeIndex: index,
        textKey: 'text',
        namespace: PINECONE_NAME_SPACE, //namespace comes from your config folder
      },
    );
    //create chain
    const pineconeChain = makePineconeChain(vectorStore);
    //Ask a question using chat history

    // const qaTool = new makePineconeChainTool({
    //   name: "my-qa-tool",
    //   description: "My ConversationalRetrievalQAChain tool",
    //   chain: pineconeChain,
    //   inputKeys: ["question", "chat_history"],
    // });

    let response = await pineconeChain.call({
      question: sanitizedQuestion,
      chat_history: history || [],
    });

    if ((response.text as string).includes('ANSWER NOT FOUND')) {
      //re call serp API chain with the same question and chat history

      const serpapiTool = new SerpAPI(process.env.SERPAPI_API_KEY, {
        location: 'Austin,Texas,United States',
        hl: 'en',
        gl: 'us',
      });

      // Create a ChatAgent instance from the ChatOpenAI model and the list of tools
      const model = new OpenAI({
        temperature: 0, // increase temepreature to get more creative answers
        modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
      });
      const chatAgent = ChatAgent.fromLLMAndTools(model, [serpapiTool]);

      // Create an AgentExecutor instance from the ChatAgent and the list of tools
      const executor = AgentExecutor.fromAgentAndTools({
        agent: chatAgent,
        tools: [serpapiTool],
        verbose: true,
      });

      // Define a chat prompt template that includes the chat history
      const chatPrompt = ChatPromptTemplate.fromPromptMessages([
        SystemMessagePromptTemplate.fromTemplate(
          `The following is a friendly conversation between a human and an AI.
          {chat_history}
          The AI is talkative and provides lots of specific details from its context. 
          If the AI does not know the answer to a question, it truthfully says it does not know.
          Always start your answer with: "Based on our conversation and Google search, "
          `,
        ),
        HumanMessagePromptTemplate.fromTemplate('{input}'),
      ]);

      const conversationChain = new ConversationChain({
        prompt: chatPrompt,
        llm: model,
      });

      // Call the AgentExecutor's run method with the user's question as input
      response = {
        data: await executor.run({
          input: sanitizedQuestion,
          chat_history: history || [],
          chain: conversationChain,
        }),
      };
    }

    console.log('response', response);
    res.status(200).json(response);
  } catch (error: any) {
    console.log('error', error);
    res.status(500).json({ error: error.message || 'Something went wrong' });
  }
}
