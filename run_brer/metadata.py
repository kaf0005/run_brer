"""Abstract class for handling all BRER metadata.

State and PairData classes inherit from this class.
"""


from abc import ABC
import json
import warnings


class MetaData(ABC):
    def __init__(self, name):
        """Construct metadata object. and give it a name.

        Parameters
        ----------
        name :
            Give your MetaData class a descriptive name.
        """
        self.__name = name
        self.__required_parameters = []
        self._metadata = {}

    @property
    def name(self):
        """Name that associates the class with a pair or a particular function
        (such as with the Plugins, which are named either 'training',
        'convergence' or 'production').

        Returns
        -------
        str
            the name of the class
        """
        return self.__name

    @name.getter
    def name(self):
        """getter for name.

        Returns
        -------
        str
        """
        return self.__name

    @name.setter
    def name(self, name):
        """setter for name.

        Parameters
        ----------
        name : str
            name of the class
        """
        self.__name = name

    def set_requirements(self, list_of_requirements: list):
        """Defines a set of required parameters for the class. This is quite
        useful for checking if there are any missing parameters before
        beginning a run.

        Parameters
        ----------
        list_of_requirements : list
            list of required parameters for the class (a list of strings)
        """
        self.__required_parameters = list_of_requirements

    def get_requirements(self):
        """Gets the set of required parameters for the class. This is quite
        useful for checking if there are any missing parameters before
        beginning a run.

        Returns
        -------
        list
            required parameters of the class
        """
        return self.__required_parameters

    def set(self, key=None, value=None, **kwargs):
        """Sets a parameter of the class. Checks whether or not the parameter
        is required and reports information about requirements. You can pass
        the key,value pairs either as a key and value or as a set of **kwargs.

        Parameters
        ----------
        key : str, optional
            parameter name, by default None
        value : any, optional
            parameter value, by default None
        """
        if key and value:
            if key not in self.__required_parameters and key != "name":
                warnings.warn("{} is not a required parameter of {}: setting anyway".format(key, self.name))
            self._metadata[key] = value
        for key, value in kwargs.items():
            if key not in self.__required_parameters and key != "name":
                warnings.warn("{} is not a required parameter of {}: setting anyway".format(key, self.name))
            self._metadata[key] = value

    def get(self, key):
        """Get the value of a parameter of the class.

        Parameters
        ----------
        key : str
            The name of the parameter you wish to get.

        Returns
        -------
        type
            The value of the parameter associated with the key.
        """
        return self._metadata[key]

    def set_from_dictionary(self, data: dict):
        """Another method that essentially performs the same thing as "set",
        but just takes a dictionary. Probably should be deprecated...

        Parameters
        ----------
        data : dict
            A dictionary of parameter names and values to set.
        """
        self.set(**data)

    def get_as_dictionary(self):
        """Get all of the current parameters of the class as a dictionary.

        Returns
        -------
        dict
            dictionary of class parameters
        """
        return self._metadata

    def get_missing_keys(self):
        """Gets all of the required parameters that have not been set.

        Returns
        -------
        list
            A list of required parameter names that have not been set.
        """
        missing = []
        for required in self.__required_parameters:
            if required not in self._metadata.keys():
                missing.append(required)
        return missing


class MultiMetaData(ABC):
    """A single class that handles multiple MetaData classes (useful when
    restraining multiple atom-atom pairs)."""

    def __init__(self):
        self._metadata_list = []
        self.__names = []

    @property
    def names(self):
        """A list of names that associate each class with a pair or a
        particular function (such as with the Plugins, which are named either
        'training', 'convergence' or 'production').

        Returns
        -------
        list
            a list of names
        """
        return self.__names

    @names.setter
    def names(self, names: list):
        """setter for names.

        Parameters
        ----------
        names : list
        """
        self.__names = names

    @names.getter
    def names(self):
        """getter for names.

        Returns
        -------
        list
            list of names

        Raises
        ------
        IndexError
            Raise IndexError if no metadata have been loaded.
        """
        if not self.__names:
            if not self._metadata_list:
                raise IndexError('Must import a list of metadata before retrieving names')
            self.__names = [metadata.name for metadata in self._metadata_list]
        return self.__names

    def add_metadata(self, metadata: MetaData):
        """Appends new MetaData object to self._metadata.

        Parameters
        ----------
        metadata : MetaData
            metadata to append
        """
        self._metadata_list.append(metadata)
        self.__names.append(metadata.name)

    def name_to_id(self, name):
        """Converts the name of one of the MetaData classes to it's associated
        list index (it's idx in self._metadata)

        Parameters
        ----------
        name : str
            the name of a MetaData class.

        Returns
        -------
        int
            the index of the MetaData class in the self._metadata list.

        Raises
        ------
        IndexError
            Raise IndexError if no metadata have been loaded.
        """
        if not self.__names:
            raise IndexError('{} is not a valid name.'.format(name))
        return self.__names.index(name)

    def id_to_name(self, id):
        """Converts the index of one of the MetaData classes to it's associated
        name.

        Parameters
        ----------
        id : int
            The index of the MetaData class in the list self._metadata

        Returns
        -------
        str
            the name of the metadata class
        """
        return self.__names[id]

    def __getitem__(self, item):
        return self._metadata_list[item]

    def get_as_single_dataset(self):
        """"""
        single_dataset = {}
        for metadata in self._metadata_list:
            single_dataset[metadata.name] = metadata.get_as_dictionary()
        return single_dataset

    def write_to_json(self, filename='state.json'):
        """Writes state to json.

        Parameters
        ----------
        filename : str
             (Default value = 'state.json')
        """
        json.dump(self.get_as_single_dataset(), open(filename, 'w'))

    def read_from_json(self, filename='state.json'):
        """Reads state from json.

        Parameters
        ----------
        filename : str
             (Default value = 'state.json')
        """
        # TODO: decide on expected behavior here if there's a pre-existing list of data. For now, overwrite
        self._metadata_list = []
        self._names = []
        data = json.load(open(filename, 'r'))
        for name, metadata in data.items():
            self.__names.append(name)
            metadata_obj = MetaData(name=name)
            metadata_obj.set_from_dictionary(metadata)
            self._metadata_list.append(metadata_obj)
