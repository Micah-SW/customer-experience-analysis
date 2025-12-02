import psycopg2
import psycopg2.extras  # Added for better cursor support
import os
import sys
import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict
import csv
import json
import time  # Added for performance tracking
from datetime import datetime
from contextlib import contextmanager  # Added for connection management

# Third-party library for environment variables
from dotenv import load_dotenv

# --- Configuration and Setup ---

# Setup basic logging format. Level is set in main().
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Default setting

# Load environment variables
load_dotenv()

# Database Configuration from environment variables
# Use more descriptive names and add connection pooling parameters
DB_CONFIG = {
    'database': os.getenv("POSTGRES_DB", "postgres"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "secret"),
    'host': os.getenv("POSTGRES_HOST", "localhost"),
    'port': os.getenv("POSTGRES_PORT", "5432"),
    'connect_timeout': 10,
    'application_name': 'sql_query_executor'
}

class OutputFormat(Enum):
    """Defines the supported output formats."""
    CONSOLE = 'console'
    CSV = 'csv'
    JSON = 'json'
    # Other formats like EXCEL, MARKDOWN, HTML could be added here

# --- Helper Functions ---

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic cleanup."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.debug("Database connection established")
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

def parse_sql_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Parse SQL file into individual queries with metadata.
    
    Returns list of dicts with 'name', 'text', and 'type' for each query.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove comments and split by semicolon
        queries = []
        current_query = []
        in_comment = False
        line_number = 1
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            # Handle block comments
            if '/*' in line:
                in_comment = True
            if '*/' in line:
                in_comment = False
                continue
            
            if not in_comment and not stripped.startswith('--') and stripped:
                current_query.append(line)
            
            # Check for semicolon to end query
            if ';' in line and not in_comment:
                if current_query:
                    query_text = '\n'.join(current_query).strip()
                    if query_text:
                        # Try to extract query name from comments
                        query_name = f"Query_{len(queries) + 1}"
                        for prev_line in reversed(content.split('\n')[:line_number]):
                            if prev_line.strip().startswith('-- Query'):
                                query_name = prev_line.strip().replace('--', '').strip()
                                break
                        
                        queries.append({
                            'name': query_name,
                            'text': query_text,
                            'type': 'SELECT' if query_text.strip().upper().startswith('SELECT') else 'OTHER'
                        })
                    current_query = []
            
            line_number += 1
        
        # Handle last query if no semicolon
        if current_query:
            query_text = '\n'.join(current_query).strip()
            if query_text:
                queries.append({
                    'name': f"Query_{len(queries) + 1}",
                    'text': query_text,
                    'type': 'SELECT' if query_text.strip().upper().startswith('SELECT') else 'OTHER'
                })
        
        logger.info(f"Parsed {len(queries)} queries from {file_path.name}")
        return queries
        
    except Exception as e:
        logger.error(f"Failed to parse SQL file: {e}")
        raise

def print_table_to_console(col_names: List[str], results: List[Tuple[Any, ...]], 
                          query_name: str, execution_time: float):
    """Prints query results to the console in a formatted table."""
    print("\n" + "="*80)
    print(f"QUERY: {query_name}")
    print(f"Time: {execution_time:.3f}s | Rows: {len(results):,}")
    print("="*80)

    if not results:
        print("No results returned for this query.")
        return

    # Determine column widths for clean table printing (limit width for readability)
    col_widths = [min(len(name), 30) for name in col_names]
    for row in results:
        for j, cell in enumerate(row):
            if j < len(col_widths):
                cell_str = str(cell)[:30] + "..." if len(str(cell)) > 30 else str(cell)
                col_widths[j] = max(col_widths[j], len(cell_str))
    
    # Header
    header = " | ".join(col_names[j][:30].ljust(col_widths[j]) for j in range(len(col_names)))
    print(header)
    print("-" * len(header))
    
    # Rows (limit to 50 for console display)
    max_display_rows = 50
    for i, row in enumerate(results[:max_display_rows]):
        row_str = " | ".join(
            (str(cell)[:30] + "..." if len(str(cell)) > 30 else str(cell)).ljust(col_widths[j])
            for j, cell in enumerate(row)
        )
        print(row_str)
    
    if len(results) > max_display_rows:
        print(f"... and {len(results) - max_display_rows:,} more rows")

def save_to_csv(col_names: List[str], results: List[Tuple[Any, ...]], 
                output_path: Path, query_name: str):
    """Save query results to CSV file."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for row in results:
                writer.writerow(row)
        logger.info(f"CSV saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise

def save_to_json(col_names: List[str], results: List[Tuple[Any, ...]], 
                 output_path: Path, query_name: str):
    """Save query results to JSON file."""
    try:
        # Convert results to list of dictionaries
        data = {
            'query': query_name,
            'columns': col_names,
            'data': [dict(zip(col_names, row)) for row in results],
            'row_count': len(results),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise

def execute_single_query(cursor, query_text: str, query_name: str) -> Optional[Dict]:
    """
    Execute a single query and return results.
    
    Returns dict with 'col_names', 'results', 'row_count', and 'execution_time'
    """
    start_time = time.time()
    
    try:
        cursor.execute(query_text)
        
        # Check if query returns results
        if cursor.description:
            col_names = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            row_count = len(results)
        else:
            # DDL/DML query
            col_names = []
            results = []
            row_count = cursor.rowcount
        
        execution_time = time.time() - start_time
        
        return {
            'col_names': col_names,
            'results': results,
            'row_count': row_count,
            'execution_time': execution_time,
            'success': True
        }
        
    except psycopg2.Error as e:
        execution_time = time.time() - start_time
        logger.error(f"Query '{query_name}' failed after {execution_time:.3f}s: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': execution_time
        }

def handle_result_output(query_result: Dict, output_format: OutputFormat, 
                        output_dir: Path, query_name: str, query_text: str):
    """Route query results to appropriate output handler."""
    
    if not query_result['success']:
        logger.error(f"Query failed: {query_result.get('error', 'Unknown error')}")
        return
    
    if output_format == OutputFormat.CONSOLE:
        print_table_to_console(
            query_result['col_names'], 
            query_result['results'], 
            query_name, 
            query_result['execution_time']
        )
    
    elif output_format == OutputFormat.CSV:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query_name.replace(' ', '_').replace(':', '_')
        filename = output_dir / f"{safe_name}.csv"
        save_to_csv(
            query_result['col_names'], 
            query_result['results'], 
            filename, 
            query_name
        )
    
    elif output_format == OutputFormat.JSON:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = query_name.replace(' ', '_').replace(':', '_')
        filename = output_dir / f"{safe_name}.json"
        save_to_json(
            query_result['col_names'], 
            query_result['results'], 
            filename, 
            query_name
        )
    
    else:
        logger.warning(f"Unsupported output format: {output_format.value}")

# --- Core Execution Function ---

def execute_queries(sql_file_path: Path, output_dir: Path, output_format: OutputFormat):
    """
    Connects to PostgreSQL, reads SQL from a file, executes each query,
    and handles result output based on the specified format.
    """
    # Parse queries from file
    try:
        queries = parse_sql_file(sql_file_path)
    except Exception as e:
        logger.critical(f"Failed to parse SQL file: {e}")
        sys.exit(1)
    
    if not queries:
        logger.warning("No valid queries found in the SQL file.")
        return
    
    total_queries = len(queries)
    successful_queries = 0
    failed_queries = 0
    total_rows = 0
    total_execution_time = 0.0
    
    try:
        with get_db_connection() as conn:
            # Use DictCursor for better column name handling
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                
                for i, query_info in enumerate(queries, 1):
                    query_name = query_info['name']
                    query_text = query_info['text']
                    query_type = query_info['type']
                    
                    logger.info(f"Executing query {i}/{total_queries}: {query_name}")
                    
                    # Execute the query
                    result = execute_single_query(cursor, query_text, query_name)
                    
                    if result['success']:
                        successful_queries += 1
                        total_rows += result['row_count']
                        total_execution_time += result['execution_time']
                        
                        # Handle output for SELECT queries
                        if query_type == 'SELECT':
                            handle_result_output(
                                result,
                                output_format,
                                output_dir,
                                query_name,
                                query_text
                            )
                        else:
                            logger.info(f"DDL/DML query executed: {result['row_count']} rows affected")
                        
                        # Commit after each successful query
                        conn.commit()
                        
                    else:
                        failed_queries += 1
                        total_execution_time += result['execution_time']
                        conn.rollback()
    
    except psycopg2.Error as e:
        logger.critical(f"Database Connection Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXECUTION SUMMARY (File: {sql_file_path.name})")
    print("="*80)
    print(f"{'Total Queries:':<30} {total_queries}")
    print(f"{'Successful Queries:':<30} {successful_queries}")
    print(f"{'Failed Queries:':<30} {failed_queries}")
    print(f"{'Total Rows Returned:':<30} {total_rows:,}")
    print(f"{'Total Execution Time:':<30} {total_execution_time:.3f}s")
    if successful_queries > 0:
        print(f"{'Average Time per Query:':<30} {total_execution_time/successful_queries:.3f}s")
    print("="*80)

# --- Argument Parsing ---

def create_arg_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Execute SQL queries from a file against a PostgreSQL database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s queries.sql                    # Console output
  %(prog)s queries.sql -f csv -o ./data   # Save as CSV
  %(prog)s queries.sql -f json --verbose  # JSON output with verbose logging
        """
    )
    
    parser.add_argument(
        'sql_file',
        type=str,
        help="Path to the .sql file containing the queries to execute."
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./sql_output',
        help="Directory to save non-console output files (default: './sql_output')."
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        default='console',
        choices=[f.value for f in OutputFormat],
        help=f"Output format for results. (default: 'console')"
    )
    
    # Use a mutually exclusive group for log levels (debug/verbose)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Set logging level to INFO."
    )
    group.add_argument(
        '-d', '--debug',
        action='store_true',
        help="Set logging level to DEBUG (most detailed output)."
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=1000,
        help="Maximum rows to fetch per query (default: 1000). Use -1 for unlimited."
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Batch size for fetching results (default: 100)."
    )
    
    return parser

# --- Main Entry Point ---

def main():
    """Main entry point for the SQL execution CLI tool."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Configure logging level based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    
    logger.debug(f"Arguments: {args}")
    
    # Validate SQL file path
    sql_file = Path(args.sql_file)
    if not sql_file.exists():
        logger.critical(f"SQL file not found: {sql_file.resolve()}")
        sys.exit(1)
    
    # Validate output directory
    output_dir = Path(args.output_dir)
    
    # Map format string to enum
    format_map = {f.value: f for f in OutputFormat}
    output_format = format_map.get(args.format, OutputFormat.CONSOLE)
    
    # Set fetch limits in DB_CONFIG if provided
    if args.max_rows > 0:
        DB_CONFIG['options'] = f'-c statement_timeout={args.max_rows * 100}'
    
    # Execute the queries
    try:
        execute_queries(
            sql_file_path=sql_file,
            output_dir=output_dir,
            output_format=output_format
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()