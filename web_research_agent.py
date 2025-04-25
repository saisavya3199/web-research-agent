import os

os.environ["GOOGLE_API_KEY"] = ""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
)

# defining search tool which uses LLM to return the source links based on prompt defined below
@tool
def search_tool(query: str):
  """tool to search for source links"""
  prompt = f"""
You are a helpful assistant designed to extract only the most relevant and trustworthy source links for user queries.

Your task:
1. Understand the **intent** behind the following query.
2. Break down complex questions into effective sub-queries if needed.
3. Identify the **type of information** needed: factual, analytical, opinion-based, recent news, or historical context.
4. Retrieve source URLs that are:
   - Relevant to the user's intent
   - From **trustworthy and reputable domains** 
   - Not promotional, AI-generated spam, or low-quality

Query:
"{query}"

Output format:
- Return only the final list of reliable URLs (one per line)
- Do not include titles, commentary, or explanation
- Do not include any introductory or trailing text

Example:
https://www.nytimes.com/example
https://www.nature.com/article/example
https://www.whitehouse.gov/briefing/example
"""

  raw = llm.invoke(prompt)
  if isinstance(raw, dict) and "content" in raw:
      raw = raw["content"]  # Fix if llm returns object

  urls = [line.strip() for line in raw.content.splitlines() if line.strip().startswith("http")]
  print("Raw LLM Output:\n", raw)
  print("Parsed URLs:\n", urls)
  return urls

# defining crawl tool which uses Crawl4AI to crawl the source links and return the markdown
# PruningContentFilter is used to filter out the markdown keeping only the relevant content
@tool
async def crawl_urls_tool(urls):
  """tool to crawl source links"""
  config = CrawlerRunConfig(
      markdown_generator=DefaultMarkdownGenerator(
          content_filter=PruningContentFilter(threshold=0.6),
          options={"ignore_links": True}
      )
  )
  if isinstance(urls, str):
        urls = [urls]
  valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]

  async with AsyncWebCrawler() as crawler:
      for url in valid_urls:
        if not url.startswith(('http://', 'https://')):
          print(f"Invalid URL: {url}")
        else:
          print(f"\n Crawling: {url}")
          result = await crawler.arun(url, config=config)
          if result.success:
              print("Filtered markdown:\n", result.markdown.fit_markdown)
          else:
              print("Crawl failed:", result.error_message)

#defining summarize tool that will summarize the markdown content using LLM based on the prompt defined below
@tool
def summarize_markdown_tool(markdown: str) -> str:
    """
    Summarizes the given markdown content into concise key points using an LLM.
    Input should be markdown content scraped from a webpage.
    """
    prompt = f"""
You are a helpful assistant. Summarize the following markdown content into concise bullet points. Focus on the key ideas and important takeaways.

Markdown Content:
\"\"\"
{markdown}
\"\"\"

Output format:
- Point 1
- Point 2
- ...
"""
    response = llm.invoke(prompt)
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    return response

tools=[summarize_markdown_tool, crawl_urls_tool, search_tool]

# defining react agent which acts as the web research agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = asyncio.run(agent.ainvoke({"input": "What is the recent news in Kashmir?"}))
print(result)