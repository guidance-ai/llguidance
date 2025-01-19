import pandas as pd
import json
import yaml
import argparse

def load_frame(path):
    # Load the JSON file
    with open(path, "r") as file:
        data = json.load(file)

    # Normalize the JSON data
    rows = []
    for category_index, category in enumerate(data):
        category_name = category["category"]
        for case_index, case in enumerate(category["cases"]):
            for test_index, test in enumerate(case["tests"]):
                rows.append({
                    "category": category_name,
                    "case": case_index,
                    "test": test_index,
                    "valid": test["valid"],
                    "success": test["success"],
                })

    # Create the DataFrame
    df = pd.DataFrame(rows)

    # Set the MultiIndex
    df.set_index(["category", "case", "test"], inplace=True)
    
    return df


def load_frames(baseline_path, test_path):
    # Load the DataFrames
    df = pd.concat({
        "baseline": load_frame(baseline_path),
        "test": load_frame(test_path),
    }, axis=1)

    return df

def diff(baseline_path, test_path):
    df = load_frames(baseline_path, test_path)
    assert df[('baseline', 'valid')].equals(df[('test', 'valid')])

    # Identify regressed and improved tests
    df['regressed'] = df[("baseline", "success")] & ~df[("test", "success")]
    df['improved'] = ~df[("baseline", "success")] & df[("test", "success")]

    # Build the result dictionary
    regressions = {}
    improvements = {}
    for category, category_data in df.groupby(level='category'):
        cat_reg = []
        cat_imp = []
        
        for (case, case_data) in category_data.groupby(level='case'):
            regressed_tests = case_data[case_data['regressed']].index.get_level_values('test').tolist()
            improved_tests = case_data[case_data['improved']].index.get_level_values('test').tolist()
            
            if regressed_tests:
                cat_reg.append({"case": case, "tests": regressed_tests})
            if improved_tests:
                cat_imp.append({"case": case, "tests": improved_tests})
        
        if cat_reg:
            regressions[category] = cat_reg
        if cat_imp:
            improvements[category] = cat_imp

    result = {"regressions": regressions, "improvements": improvements}
    return result

def main(baseline_path, test_path):
    delta = diff(baseline_path, test_path)
    print(yaml.dump(delta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_path", type=str, help="Path to the baseline JSON file")
    parser.add_argument("test_path", type=str, help="Path to the test JSON file")
    args = parser.parse_args()
    main(args.baseline_path, args.test_path)
