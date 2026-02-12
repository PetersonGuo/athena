"""Example script with a subtle bug for testing the debugger.

Run with: athena examples/simple_bug.py
"""


def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    count = 0
    for num in numbers:
        total += num
        count += 1
    return total / count


def process_data(data):
    """Process a dataset and return statistics."""
    results = {}
    for key, values in data.items():
        avg = calculate_average(values)
        results[key] = {
            "average": avg,
            "count": len(values),
            "total": sum(values),
        }
    return results


def main():
    data = {
        "temperatures": [72, 68, 75, 80, 65],
        "humidity": [45, 50, 55, 60, 40],
        "wind_speed": [],
    }

    print("Processing weather data...")
    stats = process_data(data)

    for key, value in stats.items():
        print(f"  {key}: avg={value['average']:.1f}, count={value['count']}")


if __name__ == "__main__":
    main()
