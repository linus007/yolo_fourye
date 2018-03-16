import configparser
DEBUG = True
def process_config(conf_file):
    """
    Arg:
        conf_file configure file path
    Returns:
        DataSetParams,
    """
    # param sets
    dataset_conf = {}

    # common sets
    common_conf = {}

    # net sets
    net_conf = {}

    # solver sets
    solver_conf = {}

    conf_parser = configparser.ConfigParser()
    conf_parser.read(conf_file)


    for section in conf_parser.sections():
        # construct dataset params
        if section == "DataSet":
            for option in conf_parser.options(section):
                dataset_conf[option] = conf_parser.get(section, option)
        elif section == "Common":
            for option in conf_parser.options(section):
                common_conf[option] = conf_parser.get(section, option)
        elif section == 'Net':
            for option in conf_parser.options(section):
                net_conf[option] = conf_parser.get(section, option)
        elif section == 'Solver':
            for option in conf_parser.options(section):
                solver_conf[option] = conf_parser.get(section, option)


    if DEBUG:
        print("dataset_conf:")
        for k, v in dataset_conf.items():
            print("    {}: {}".format(k, v))
        print("")

        print("common_conf:")
        for k, v in common_conf.items():
            print("    {}: {}".format(k, v))
        print("")


        print("net_conf:")
        for k, v in net_conf.items():
            print("    {}: {}".format(k, v))
        print("")

        print("solver_conf:")
        for k, v in solver_conf.items():
            print("    {}: {}".format(k, v))
        print("")

    return dataset_conf, common_conf, net_conf, solver_conf
