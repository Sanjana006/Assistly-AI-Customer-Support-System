from graph.graph import process_ticket
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

console = Console()

def demo():
    # Test cases covering all agent paths
    test_cases = [
        {
            "message": "Hi, where is my order ORD00042? It was supposed to arrive 3 days ago.",
            "label": "Order Status (neutral)"
        },
        {
            "message": "THIS IS RIDICULOUS!! I ordered ORD00015 TWO WEEKS AGO and it still hasnt arrived! I WANT A FULL REFUND NOW!!",
            "label": "Refund Request (angry)"
        },
        {
            "message": "What is your return policy for electronics?",
            "label": "General Policy (neutral)"
        },
        {
            "message": "I want to talk to a real person please.",
            "label": "Human Request (direct escalation)"
        }
    ]
    
    for test in test_cases:
        console.print(f"\n[bold yellow]TEST: {test['label']}[/bold yellow]")
        console.print(f"[dim]Message: {test['message'][:100]}[/dim]\n")
        
        result = process_ticket(test["message"])
        
        # Display results in a nice table
        table = Table(show_header=False, box=None, padding=(0,1))
        table.add_column("Key", style="cyan", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("Intent", result.get('intent', 'N/A'))
        table.add_row("Sentiment", result.get('sentiment', 'N/A'))
        table.add_row("Frustration", f"{result.get('frustration_score', 0):.2f}")
        table.add_row("Tools Used", str([t['tool'] for t in result.get('tools_called', [])]))
        table.add_row("QA Score", f"{result.get('quality_score', 0):.2f}")
        table.add_row("Escalated", str(result.get('needs_escalation', False)))
        table.add_row("Time", f"{result.get('processing_time_ms', 0):.0f}ms")
        
        console.print(table)
        console.print(Panel(
            result.get('final_response', 'No response generated'),
            title="[green]Final Response[/green]",
            border_style="green"
        ))

if __name__ == "__main__":
    demo()