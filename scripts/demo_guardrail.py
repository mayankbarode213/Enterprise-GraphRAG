import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.agents.guardrail import GuardrailAgent
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit(
        "[bold red]Pydantic v2 Guardrail Validation Demo[/bold red]\n"
        "[dim]Triggering demonstrate_validation_failure() to catch invalid LLM outputs[/dim]",
        border_style="red"
    ))

    agent = GuardrailAgent()

    try:
        agent.demonstrate_validation_failure()
    except ValidationError as exc:
        console.print("\n[bold green][PASS] Success: ValidationError caught as expected![/bold green]\n")
        console.print(f"[bold]Total errors caught:[/bold] {exc.error_count()}\n")
        
        for err in exc.errors():
            field_name = " -> ".join(str(item) for item in err['loc'])
            error_message = err['msg']
            input_value = err.get('input')
            
            console.print(f"[bold red][FAIL] Field Error:[/bold red] [cyan]{field_name}[/cyan]")
            console.print(f"   [yellow]Rule violation:[/yellow] {error_message}")
            console.print(f"   [yellow]Rejected input:[/yellow] {repr(input_value)}")
            console.print("—" * 60)

if __name__ == "__main__":
    main()
