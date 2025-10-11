# API Based Operations

- You need to create a `config.json` file under the specified `CONFIG_PATH` to store authentication information. The default value is `CONFIG_PATH = "./api_task/config.json"`.

    - For the specific format, see [config template](config_template.json).

- For details about Bilibili authentication information, refer to the documentation: [Credential](https://nemo2011.github.io/bilibili-api/#/get-credential)

    - All Bilibili-related code uses **asynchronous operations**. Since web scraping is involved, excessive high concurrency may lead to account bans.

## Usage

Currently supported API operations:

- Automatic email sending:

    - Supports attaching local files

    - Supports attaching files from Android phones using adb shell

    - Supports binary files such as photos and videos

- Bilibili-related operations:

    - Supports like/unlike, coin, and one-click triple actions

    - Supports video search, crawling user information, and video information

## Demo

See [API_Demo](../test/api_test.py) for more information.