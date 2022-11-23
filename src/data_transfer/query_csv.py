""" Helper class to load database-dl.xlsx and
query it based on defined query in config.py"""
import pandas as pd
import numpy as np
from py_topping.data_connection.sharepoint import lazy_SP365


class QueryCSV:
    """
    A class that contains methods
    for automatically downloading database_dl

    ...

    Attributes
    ----------
    credentials: credentials for sharepoint
        see config.yaml
    excel_file: string
        path/filename to store database-dl.xlsx


    Methods
    -------
    query(query)
        Filters out desired patients from database-dl
    sharepoint_download(credentials, excel_file)
        Downloads database-dl
    """

    def __init__(self, credentials, excel_file):
        """
        Parameters
        ----------
        credentials: credentials for sharepoint
            see config.yaml
        excel_file: string
            path/filename to store database-dl.xlsx
        """
        # download database-dl
        self.sharepoint_download(credentials, excel_file)
        database_dl = pd.read_excel(excel_file)
        self.data = database_dl.replace(np.nan, "None")

    def query(self, query):
        """
        Parameters
        ----------
        query: string
            Filters out specified patients
        """
        self.data = self.data.query(query)
        #self.data = self.data[-6:]

        return list(
            zip(
                self.data["PatientID"],
                self.data["Series"],
                self.data["Timestamp"],
                self.data["Cropped"],
            )
        )

    @staticmethod
    def sharepoint_download(credentials, excel_file):
        """
        Parameters
        ----------
        credentials: credentials for sharepoint
            see config.yaml
        excel_file: string
            path/filename to store database-dl.xlsx
        """
        sharepoint = lazy_SP365(
            site_url=credentials["url"],
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
        )

        sharepoint.download(
            sharepoint_location=credentials["location"], local_location=excel_file
        )
