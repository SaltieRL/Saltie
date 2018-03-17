import json
import os
import io
import random
import requests
import zipfile
from bot_code.conversions import window_converter


class ServerConverter:
    file_status = {}
    local_file = None
    config_response = None
    download_config = False
    download_model = False
    error = False

    def __init__(self, server_ip, uploading, download_config, download_model,
                 num_players=2, num_my_team=1, username='', model_hash='', is_eval=False):
        self.server_ip = server_ip
        self.uploading = uploading
        self.download_config = download_config
        self.download_model = download_model
        self.username = username
        self.num_players = num_players
        self.num_my_team = num_my_team
        self.model_hash = model_hash
        self.is_eval = is_eval
        self.ping_server()

    def set_player_username(self, username):
        print('setting username', username)
        self.username = username

    def set_player_amount(self, num_players, num_my_team):
        print('setting players', num_players)
        print('num on my team', num_my_team)
        self.num_players = num_players
        self.num_my_team = num_my_team

    def set_model_hash(self, model_hash):
        print('setting model hash', model_hash)
        self.model_hash = str(model_hash)

    def set_is_eval(self, is_eval):
        print('setting is eval', is_eval)
        self.is_eval = is_eval

    def load_config(self):
        """
        Makes a request to download the config from the server.
        Times out after 10 seconds.
        :return: None
        """
        if self.download_config:
            print('downloading config')
            try:
                self.config_response = requests.get(self.server_ip + '/config/get', timeout=10)
                print('config downloaded')
            except Exception as e:
                print('Error downloading config, reverting to file on disk:', e)
                self.download_config = False

    def load_model(self, model_hash):
        """
        Makes a request to download the config from the server.
        Times out after 10 seconds.
        Unzips the downloaded zip file
        and saves the files in a folder called training/saltie
        :return: None
        """
        if self.download_model:
            print('downloading model')
            folder = os.path.join('training', 'saltie', model_hash)
            try:
                url = self.server_ip + '/model/get/' + model_hash
                print(url)
                r = requests.get(url, timeout=10, stream=True)
                try:
                    if not r.json():
                        return
                except:
                    pass  # this is a file
                print('model downloaded')
                in_memory_file = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        in_memory_file.write(chunk)
                print('downloaded model')
                with zipfile.ZipFile(in_memory_file) as f:
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                    for file in f.namelist():
                        contents = f.open(file)
                        print(file)
                        with open(os.path.join(folder, os.path.basename(file)), "wb") as unzipped:
                            unzipped.write(contents.read())
            except Exception as e:
                print('Error downloading model, not writing it:', e)
                download_model = False

    def maybe_upload_replay(self, fn, model_hash):
        if not self.uploading:
            self.add_to_local_files(fn)
            return

        try:
            self._upload_replay(fn, model_hash)
        except Exception as e:
            print('catching all errors to keep the program going', e)

    def _upload_replay(self, fn, model_hash):
        with open(fn, 'rb') as f:
            try:
                self._upload_replay_opened_file(f, model_hash)
                self.file_status[fn] = True
            except ConnectionRefusedError as error:
                print('server is down ', error)
                self.add_to_local_files(fn)
            except ConnectionError as error:
                print('server is down', error)
                self.add_to_local_files(fn)
            except Exception as e:
                print('server is down, general error', e)
                self.add_to_local_files(fn)

    def _upload_replay_opened_file(self, file, model_hash):
        payload = {'username': self.username, 'hash': model_hash,
                   'num_my_team': self.num_my_team, 'num_players': self.num_players, 'is_eval': self.is_eval}
        r = requests.post(self.server_ip + '/upload/replay', files={'file': file}, data=payload, timeout=10)
        if r.status_code != 200 and r.status_code != 202:
            print('i=something went wrong in the server ', r.status_code)
            print(r.content)
        else:
            print('Upload:', r.json()['status'])

    def add_to_local_files(self, fn):
        """
        Adds this file to a list of files to upload at a later time.
        :param fn: the file that failed to upload
        :return: None
        """
        if fn not in self.file_status:
            self.file_status[fn] = False

    def retry_files(self):
        """
        Try reuploading any files that did not upload the first time.
        :return: None
        """
        for key in self.file_status:
            if not self.file_status[key]:
                print('retrying file:', key)
                self.maybe_upload_replay(key, self.model_hash)
        print('all files retried')

    def download_files(self):
        self.load_config()
        self.load_model()

    def get_replays(self, num_replays, get_eval_only):
        r = requests.get(self.server_ip + '/replays/list')
        replays = r.json()
        print('num replays available', len(replays), ' num requested ', num_replays)
        n = min(num_replays, len(replays))
        return [";".join(random.sample(replays, n))]

    def download_file(self, file):
        if file.contains(';'):
            file = file.split(';')
            response = requests.post(self.server_ip + '/replays/download', json=json.dumps({'files': file}))
            zip = zipfile.ZipFile(response.content)
            return [zip.read(name) for name in zip.namelist()]
        else:
            response = requests.get(self.server_ip + '/replays/' + file)
            return io.BytesIO(response.content)

    def ping_server(self):
        try:
            response = requests.head(self.server_ip, timeout=10)
            if response.status_code != 200 and response.status_code != 202:
                self.Error = True
        except Exception as e:
            self.Error = True
        if self.server_ip.endswith('/'):
            self.warn_server('Server IP Ends with / when it should not')
            self.Error = True

    def warn_server(self, issue_string):
        print(issue_string)
        window_converter.create_popup(issue_string)
