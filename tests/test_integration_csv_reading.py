import helmholtz_firedrake.utils as hhu

def test_basic_csv():
    """Basically checks that nothing goes wrong. Not a true integration
    test, as it doesn't compare with the latest version of code that
    generates csvs (see gen.py).
    """

    csv_list = ['no-1-2018-12-07T16:13:10.236634.csv','no-2-2018-12-07T16:13:10.248309.csv']

    names_list = ['k','blarg']

    output_df = hhu.csv_list_to_dataframe(csv_list,names_list)