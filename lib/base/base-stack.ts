/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3'
import * as ssm from 'aws-cdk-lib/aws-ssm'
import { Construct } from 'constructs';

export interface StackCommonProps extends cdk.StackProps {
    projectPrefix: string;
    appConfig: any;
}

export class BaseStack extends cdk.Stack {
    protected projectPrefix: string;
    protected commonProps: StackCommonProps;
    protected stackConfig: any;

    constructor(scope: Construct, id: string, commonProps: StackCommonProps, stackConfig: any) {
        super(scope, id, commonProps);

        this.projectPrefix = commonProps.projectPrefix;
        this.commonProps = commonProps;
        this.stackConfig = stackConfig;
    }

    protected createS3Bucket(baseName: string): s3.Bucket {
        const suffix: string = `${this.commonProps.env?.region}-${this.commonProps.env?.account?.substr(0, 4)}`
        const timestamp = new Date().getTime();

        const s3Bucket = new s3.Bucket(this, baseName, {
            bucketName: `${this.projectPrefix}-${baseName}-${suffix}`.toLowerCase().replace('_', '-'),
            versioned: false,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true
        });

        return s3Bucket;
    }


    protected putParameter(paramKey: string, paramValue: string): void {
        const paramKeyWithPrefix = `${this.projectPrefix}-${paramKey}`;

        new ssm.StringParameter(this, paramKey, {
            parameterName: paramKeyWithPrefix,
            stringValue: paramValue,
            description: `Parameter for ${paramKey}`,
            dataType: ssm.ParameterDataType.TEXT,
            tier: ssm.ParameterTier.STANDARD
        });
    }

    protected getParameter(paramKey: string): string {
        const paramKeyWithPrefix = `${this.projectPrefix}-${paramKey}`;

        return ssm.StringParameter.valueForStringParameter(
            this,
            paramKeyWithPrefix
        );
    }
}

