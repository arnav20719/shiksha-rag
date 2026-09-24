import json
from langchain_core.documents import Document

def load_colleges(path: str) -> list[dict]:
    """Load colleges from the JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]["colleges"]

def college_to_document(college: dict) -> Document:
    """Convert a single college dict into a LangChain Document."""
    # Handle missing values gracefully
    salary = college.get("median_salary") or "Not available"
    rating = college.get("rating") or "Not rated"
    ownership = college.get("ownership") or "Not specified"
    
    text = f"""
College: {college['name']}
Location: {college['location']}
Ownership: {ownership}
Rating: {rating} out of 5
Total Courses Offered: {college['course_count']}
Median Salary Range: {salary}
Placement above 8 LPA: {"Yes" if college.get("placement_above_8lpa") else "No"}
""".strip()
    
    return Document(
        page_content=text,
        metadata={
            "college_name": college["name"],
            "location": college["location"],
            "rating": college.get("rating"),
            "course_count": college.get("course_count"),
            "source_url": college.get("url"),
            "type": "college_overview"
        }
    )

def build_all_documents(path: str) -> list[Document]:
    """Load all colleges and convert each to a Document."""
    colleges = load_colleges(path)
    return [college_to_document(c) for c in colleges]

if __name__ == "__main__":
    docs = build_all_documents("data/iit_colleges.json")
    print(f"Total documents created: {len(docs)}")
    print("\n--- First document preview ---")
    print("Content:")
    print(docs[0].page_content)
    print("\nMetadata:")
    print(docs[0].metadata) 
