import os
import io
import random
import requests
import zipfile


class ServerConverter:
    file_status = {}
    local_file = None
    config_response = None
    download_config = False
    download_model = False

    def __init__(self, server_ip, uploading, download_config, download_model,
                 num_players=2, num_my_team=1, username='', model_hash=''):
        self.server_ip = server_ip
        self.uploading = uploading
        self.download_config = download_config
        self.download_model = download_model
        self.username = username
        self.num_players = num_players
        self.num_my_team = num_my_team
        self.model_hash = model_hash

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

    def load_config(self):
        if self.download_config:
            try:
                self.config_response = requests.get(self.server_ip + '/config/get')
            except Exception as e:
                print('Error downloading config, reverting to file on disk:', e)
                self.download_config = False

    def load_model(self):
        if self.download_model:
            folder = 'training\\saltie\\'
            try:
                b = requests.get(self.server_ip + '/model/get')
                bytes = io.BytesIO()
                for chunk in b.iter_content(chunk_size=1024):
                    if chunk:
                        bytes.write(chunk)
                print('downloaded model')
                with zipfile.ZipFile(bytes) as f:
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

    def maybe_upload_replay(self, fn):
        if not self.uploading:
            self.add_to_local_files(fn)
            return

        try:
            self._upload_replay(fn)
        except Exception as e:
            print('catching all errors to keep the program going', e)

    def _upload_replay(self, fn):
        with open(fn, 'rb') as f:
            try:
                self._upload_replay_opened_file(f)
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

    def _upload_replay_opened_file(self, file):
        payload = {'username': self.username, 'hash': str(self.model_hash),
                   'num_my_team': self.num_my_team, 'num_players': self.num_players}
        r = requests.post(self.server_ip, files={'file': file}, data=payload)
        if r.status_code != 200 and r.status_code != 202:
            print('i=something went wrong in the server ', r.status_code)
            print(r.content)
        else:
            print('Upload:', r.json()['status'])

    def add_to_local_files(self, fn):
        if fn not in self.file_status:
            self.file_status[fn] = False

    def retry_files(self):
        for key in self.file_status:
            if not self.file_status[key]:
                print('retrying file:', key)
                self.maybe_upload_replay(key)
        print('all files retried')

    def download_files(self):
        self.load_config()
        self.load_model()

    def get_replays(self, num_replays, get_eval_only):
        r = requests.get(self.server_ip + '/replays/list')
        replays = r.json()
        print('num replays available', len(replays), ' num requested ', num_replays)
        n = min(num_replays, len(replays))
        return random.sample(replays, n)

    def download_file(self, file):
        response = requests.get(self.server_ip + '/replays/' + file)
        return io.BytesIO(response.content)
