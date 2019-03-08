# Assumes this code is being run from the top level folder. Otherwise,
# add the helmholtz_firedrake folder to your PYTHONPATH
import sys
sys.path.append('.')
import helmholtz_firedrake.utils as hhu
import pytest

# I just haven't figured out how to get it to read things yet. Hence the xfail.
@pytest.mark.xfail
def test_basic_csv():
    """Basically checks that nothing goes wrong. Not a true integration
    test, as it doesn't compare with the latest version of code that
    generates csvs (see gen.py).
    """

    csv_list = ['no-1-2018-12-07T16:13:10.236634.csv','no-2-2018-12-07T16:13:10.248309.csv']

    names_list = ['k','blarg']

    output_df = hhu.csv_list_to_dataframe(csv_list,names_list)
