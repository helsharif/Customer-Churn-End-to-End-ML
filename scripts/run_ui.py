"""Start the Streamlit churn prediction UI."""

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "src/churn_ml/ui/app.py"]))
