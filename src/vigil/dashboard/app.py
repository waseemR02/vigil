import os
import httpx
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vigil.dashboard")

# Initialize FastAPI app
app = FastAPI(title="Vigil Dashboard")

# Get API base URL from environment or use default
API_BASE_URL = os.environ.get("VIGIL_API_URL", "http://localhost:8000")

# Set up templates
templates = Jinja2Templates(directory="src/vigil/dashboard/templates")

# Add custom functions and filters to Jinja2
def now():
    """Return current datetime for use in templates."""
    return datetime.now()

def format_datetime(value, format="%Y-%m-%d %H:%M"):
    """Format a datetime object or ISO string to a readable format."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return value
    if isinstance(value, datetime):
        return value.strftime(format)
    return value

# Register custom functions and filters with Jinja2
templates.env.globals["now"] = now
templates.env.globals["min"] = min  # Add min function
templates.env.globals["max"] = max  # Add max function
templates.env.filters["format_datetime"] = format_datetime

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory="src/vigil/dashboard/static"),
    name="static"
)

# Helper function to get API client
async def get_api_client():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
        yield client

def process_api_response(response, default_value=None, log_error=True):
    """Process API response and handle errors."""
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            if log_error:
                logger.error(f"Failed to parse API response: {str(e)}")
            return default_value
    else:
        if log_error:
            logger.error(f"API error (HTTP {response.status_code}): {response.text}")
        return default_value

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, client: httpx.AsyncClient = Depends(get_api_client)):
    """Render the main dashboard."""
    try:
        # Check if API is available
        try:
            # Fetch recent incidents
            recent_incidents_response = await client.get("/incidents/recent")
            recent_incidents = process_api_response(recent_incidents_response, [])
            
            # If we got a 500 error, show error message but continue with empty data
            if recent_incidents_response.status_code == 500:
                logger.error("API error when fetching incidents: %s", recent_incidents_response.text)
                api_error_message = "Error loading incidents data. The API encountered an error."
            else:
                api_error_message = None
            
            # Fetch sources
            sources_response = await client.get("/sources")
            sources = process_api_response(sources_response, [])
            
            # Fetch tags
            tags_response = await client.get("/tags")
            tags = process_api_response(tags_response, [])
            
            # Fetch all incidents for chart data
            all_incidents_response = await client.get("/incidents?per_page=100")
            all_incidents = process_api_response(all_incidents_response, {"items": [], "total": 0})
            
            # Get counts from the response data
            incidents_count = all_incidents.get("total", 0) if isinstance(all_incidents, dict) else len(recent_incidents)
            
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "recent_incidents": recent_incidents,
                    "sources": sources,
                    "tags": tags,
                    "incidents_count": incidents_count,
                    "sources_count": len(sources),
                    "tags_count": len(tags),
                    "page_title": "Dashboard",
                    "api_status": "error" if api_error_message else "connected",
                    "api_error": api_error_message
                }
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout) as e:
            logger.error(f"API connection error: {str(e)}")
            # Return template with API connection error
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "recent_incidents": [],
                    "sources": [],
                    "tags": [],
                    "incidents_count": 0,
                    "sources_count": 0,
                    "tags_count": 0,
                    "page_title": "Dashboard",
                    "api_status": "disconnected",
                    "api_error": "Cannot connect to API server. Please ensure the API is running."
                }
            )
    except Exception as e:
        logger.exception("Error rendering dashboard: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": str(e),
                "page_title": "Error"
            }
        )

@app.get("/incidents", response_class=HTMLResponse)
async def incidents_list(
    request: Request,
    page: int = 1,
    per_page: int = 20,
    source_id: Optional[str] = None,  # Changed from int to str to catch empty strings
    tag_name: Optional[str] = None,
    client: httpx.AsyncClient = Depends(get_api_client)
):
    """Render the incidents list page."""
    try:
        # Build query params - only include non-empty parameters
        params = {"page": page, "per_page": per_page}
        if source_id and source_id.strip():
            try:
                # Convert to integer for the API
                params["source_id"] = int(source_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid source_id parameter: {source_id}")
                # Don't include if it's not a valid integer
        
        if tag_name and tag_name.strip():
            params["tag_name"] = tag_name
        
        # Fetch incidents with pagination
        incidents_response = await client.get("/incidents", params=params)
        incidents_data = process_api_response(incidents_response, {"items": [], "total": 0, "page": 1, "per_page": per_page, "pages": 1})
        
        # Fetch sources and tags for filters
        sources_response = await client.get("/sources")
        sources = process_api_response(sources_response, [])
        
        tags_response = await client.get("/tags")
        tags = process_api_response(tags_response, [])
        
        return templates.TemplateResponse(
            "incidents.html",
            {
                "request": request,
                "incidents": incidents_data.get("items", []),
                "pagination": {
                    "current_page": incidents_data.get("page", 1),
                    "total_pages": incidents_data.get("pages", 1),
                    "total_items": incidents_data.get("total", 0),
                    "per_page": incidents_data.get("per_page", per_page)
                },
                "sources": sources,
                "tags": tags,
                "selected_source_id": int(source_id) if source_id and source_id.strip() and source_id.isdigit() else None,
                "selected_tag_name": tag_name,
                "page_title": "Incidents"
            }
        )
    except Exception as e:
        logger.exception("Error rendering incidents list: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": str(e),
                "page_title": "Error"
            }
        )

@app.get("/incidents/{incident_id}", response_class=HTMLResponse)
async def incident_detail(
    request: Request,
    incident_id: int,
    client: httpx.AsyncClient = Depends(get_api_client)
):
    """Render the incident detail page."""
    try:
        # Fetch incident details
        incident_response = await client.get(f"/incidents/{incident_id}")
        if incident_response.status_code == 404:
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "Incident not found",
                    "page_title": "Not Found"
                },
                status_code=404
            )
            
        incident = process_api_response(incident_response)
        if not incident:
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": "Failed to load incident data",
                    "page_title": "Error"
                }
            )
        
        return templates.TemplateResponse(
            "incident.html",
            {
                "request": request,
                "incident": incident,
                "page_title": f"Incident: {incident.get('title', 'Unknown')}"
            }
        )
    except Exception as e:
        logger.exception("Error rendering incident detail: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": str(e),
                "page_title": "Error"
            }
        )

@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: Optional[str] = None,
    client: httpx.AsyncClient = Depends(get_api_client)
):
    """Render the search page."""
    search_results = []
    
    try:
        if q:
            # Execute search query
            search_response = await client.get("/search", params={"q": q})
            search_results = process_api_response(search_response, [])
        
        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "search_query": q,
                "search_results": search_results,
                "page_title": "Search Incidents"
            }
        )
    except Exception as e:
        logger.exception("Error in search: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": str(e),
                "page_title": "Error"
            }
        )

@app.get("/error", response_class=HTMLResponse)
async def error(
    request: Request,
    message: str = "An unknown error occurred"
):
    """Render the error page."""
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_message": message,
            "page_title": "Error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
