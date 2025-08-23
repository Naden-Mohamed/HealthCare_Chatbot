from langchain_community.tools import DuckDuckGoSearchRun
def fetch_trusted_medical_webpages(query, top_n=3):
    search = DuckDuckGoSearchRun()
    custom_tool = search.bind(
        name="trusted_medical_webpages",
        description="Search for trusted medical webpages using DuckDuckGo.",
        args_schema= DuckDuckGoSearchRun(query=query, num_results=top_n,
        filters={"site": "who.int OR cdc.gov OR nih.gov OR mayo.edu"},
        ),
    )
    results = custom_tool.invoke(query=query, num_results=top_n)
    for result in results:
        print(f"Title: {result['title']}\nURL: {result['url']}\nContent Snippet: {result['snippet'][:500]}\n")
    return results
    

if __name__ == "__main__":
    query = "hyperthyroidism"   
    trusted_pages = fetch_trusted_medical_webpages(query, top_n=5)
    for page in trusted_pages:      
        page['credibility'] = 0.9  # Adding credibility field
        page['source'] = 'Trusted Medical Source'  # Adding source field
        print(f"Title: {page['title']}\nURL: {page['url']}\nContent Snippet: {page['snippet'][:500]}\n")