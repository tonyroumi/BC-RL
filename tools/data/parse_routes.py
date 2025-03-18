import os
import argparse

def count_data_points(route_path):
    """Counts the number of files in the given route directory."""
    return sum(len(files) for _, _, files in os.walk(route_path))

def parse_weather_data_index(dataset_path, output_file):
    """Parses all route directories and writes the output in the specified format."""
    with open(output_file, 'w') as f:
        for weather_folder in sorted(os.listdir(dataset_path)):
            weather_path = os.path.join(dataset_path, weather_folder)
            if os.path.isdir(weather_path) and weather_folder.startswith("weather-"):
                data_folder = os.path.join(weather_path, "data")
                if os.path.exists(data_folder) and os.path.isdir(data_folder):
                    for route_id in sorted(os.listdir(data_folder)):
                        route_path = os.path.join(data_folder, route_id)
                        if os.path.isdir(route_path):
                            num_data_points = count_data_points(route_path)
                            f.write(f"{weather_folder}/data/{route_id}/ {num_data_points}\n")

def parse_weather_data(dataset_path, output_file):
    """Parses all route directories and writes the output in the specified format."""
    with open(output_file, 'w') as f:
        for weather_folder in sorted(os.listdir(dataset_path)):
            weather_path = os.path.join(dataset_path, weather_folder)
            if os.path.isdir(weather_path) and weather_folder.startswith("weather-"):
                data_folder = os.path.join(weather_path, "data")
                if os.path.exists(data_folder) and os.path.isdir(data_folder):
                    for route_id in sorted(os.listdir(data_folder)):
                        route_path = os.path.join(data_folder, route_id)
                        if os.path.isdir(route_path):
                            f.write(f"dataset/{weather_folder}/data/{route_id}/\n")

def parse_weather_data_blocked(dataset_path, output_file):
    """Parses all route directories and writes the output in the specified format.
    Reads block_stat.txt, splits each line by space, discards the last two elements,
    and writes the resulting string to the output file."""
    with open(dataset_path, 'r') as block_file:
        with open(output_file, 'w') as out_file:
            for line in block_file:
                parts = line.strip().split(' ')
                result = ' '.join(parts[:-2])
                out_file.write(f"{result}\n")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Parse weather data routes")
    parser.add_argument("action", type=str, choices=["dataset", "dataset_index", "blocked"],
                        help="Action to perform: dataset, dataset_index, or blocked")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the arguments based on the action specified
    if args.action == "dataset":
        output_file = 'dataset.txt'
        parse_weather_data('dataset', output_file)
    elif args.action == "dataset_index":
        output_file = 'dataset_index.txt'
        parse_weather_data_index('dataset', output_file)
    elif args.action == "blocked":
        output_file = 'blocked_routes.txt'
        parse_weather_data_blocked('blocked_stat.txt', output_file)
    
    print(f"Output written to {output_file}")