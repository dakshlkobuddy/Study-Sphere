from youtubesearchpython import VideosSearch

def get_yt_video_link(query):
    """
    Fetch top 3 YouTube video links for a given query.
    
    Args:
        query (str): Search query for YouTube
    
    Returns:
        tuple: (video_titles list, video_links list)
    """
    try:
        videos_search = VideosSearch(query=query, limit=3)
        result = videos_search.result()
        
        if not result.get('result'):
            return [], []
        
        video_titles = [video['title'] for video in result['result']]
        video_links = [video['link'] for video in result['result']]
        return video_titles, video_links
    
    except Exception as e:
        print(f"Error fetching YouTube videos: {e}")
        return [], []


# Example usage - uncomment to test
# user_query = "Explain Adaptive Radiation"
# video_titles, video_links = get_yt_video_link(user_query)
# print(video_titles, video_links)