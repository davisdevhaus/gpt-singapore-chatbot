//Create a ChainTool based on makePineconeChain
import { makePineconeChain } from '@/utils/chains/makePineconeChain';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ChainTool } from 'langchain/tools';

export const makePineconeChainTool = (vectorStore: PineconeStore) =>
  new ChainTool({
    name: 'pinecone-pdf-database',
    description:
      "Retrieves answers from a vectorstore database of PDFs. If you don't find the answer from here, use another tool.",
    chain: makePineconeChain(vectorStore),
  });
